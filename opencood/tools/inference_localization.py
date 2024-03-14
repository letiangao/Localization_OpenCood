# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time

import numpy as np
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
from opencood.visualization import vis_utils_localization
import matplotlib.pyplot as plt

from collections import OrderedDict
from opencood.utils.eval_utils_localization import cal_localization_error, cal_localization_error_reverse, pose_error_model_judgement

from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    # localization modify 0322
    # add a flag, if convert the features of two agents from A,B to B,A, find the wrong localization while inference
    loca_reverse_module = True

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    loca_error_matrix = []
    loca_error_x_list = []
    loca_error_y_list = []
    loca_error_yaw_list = []
    loca_error_pos_list = []

    loca_error_matrix_reverse = []
    loca_error_x_list_reverse = []
    loca_error_y_list_reverse = []
    loca_error_yaw_list_reverse = []
    loca_error_pos_list_reverse = []


    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    ii=0
    for i, batch_data in tqdm(enumerate(data_loader)):
        # localization modify: if the scenario contains only ego CAV, which means the other CAV is out of range, skip this scenario
        if batch_data['ego']['record_len'].sum().item() < 2 * hypes['train_params'][
            'batch_size']:  # localization modify
            continue  # localization modify
        print(ii)
        if ii>1400:
            break
        ii=ii+1
        del batch_data['ego']['merged_feature_dict']
        print("distance:", batch_data['ego']['distance'])
        lidar_ego = batch_data['ego']['lidar_np_ego_not_trans']
        lidar_cav = batch_data['ego']['lidar_np_cav_not_trans']

        pose_gt = batch_data['ego']['gt_relative_pose_for_loss']
        pose_initial = batch_data['ego']['relative_pose_for_loss']







        del batch_data['ego']['lidar_np_ego_not_trans']
        del batch_data['ego']['lidar_np_cav_not_trans']
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['label_dict']['pose'] = OrderedDict()  # localization modify
            batch_data['ego']['label_dict']['pose'].update({
                'relative_pose_for_loss': batch_data['ego']['relative_pose_for_loss'],
                'gt_relative_pose_for_loss': batch_data['ego'][
                    'gt_relative_pose_for_loss']})  # localization modify

            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, pose_error_model = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset) #localization modify
            elif opt.fusion_method == 'intermediate':
                # localization modify 0322
                # add a flag, if convert the features of two agents from A,B to B,A, find the wrong localization while inference
                batch_data['ego']['loca_inference_reverse_flg'] = False
                pred_box_tensor, pred_score, gt_box_tensor, pose_error_model = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset) #localization modify
                # localization modify 0322
                # add a flag, if convert the features of two agents from A,B to B,A, find the wrong localization while inference
                if loca_reverse_module:
                    batch_data['ego']['loca_inference_reverse_flg'] = True
                    pred_box_tensor_reverse, pred_score_reverse, gt_box_tensor_reverse, pose_error_model_reverse = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                      model,
                                                                      opencood_dataset)  # localization modify

            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)

            loca_error_matrix, loc_error_x_mean, loc_error_y_mean, loc_error_yaw_mean, loc_error_mean = cal_localization_error(loca_error_matrix,loca_error_x_list,loca_error_y_list,loca_error_yaw_list,loca_error_pos_list, pose_gt, pose_initial, pose_error_model)

            # localization modify 0322
            # add a flag, if convert the features of two agents from A,B to B,A, find the wrong localization while inference
            if loca_reverse_module:
                pose_error_model_flg = pose_error_model_judgement(pose_error_model, pose_error_model_reverse)
                if pose_error_model_flg:
                    loca_error_matrix_reverse, loc_error_x_mean_reverse, loc_error_y_mean_reverse, loc_error_yaw_mean_reverse, loc_error_mean_reverse = cal_localization_error_reverse(
                        loca_error_matrix_reverse, loca_error_x_list_reverse, loca_error_y_list_reverse, loca_error_yaw_list_reverse,
                        loca_error_pos_list_reverse, pose_gt, pose_initial, pose_error_model)


            print('mean_err:pos', loc_error_mean, ', yaw:', loc_error_yaw_mean, ', x:', loc_error_x_mean, ', y:', loc_error_y_mean)

            # localization modify: create pcd for localization visualization
            pcd_loca_visual = vis_utils_localization.transLidarToPcd(lidar_ego, lidar_cav, pose_gt, pose_initial,
                                                                     pose_error_model,
                                                                     hypes['preprocess']['cav_lidar_range'])

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)



            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  pcd_loca_visual,#batch_data['ego'][
                                                      #'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)


            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        pcd_loca_visual, #batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='intensity'#mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)
    # save localization results to txt
    noise = 100*hypes['wild_setting']['xyz_std']+hypes['wild_setting']['ryp_std']
    f = open(os.path.join(saved_path, 'error_pos_noise%d.txt' % noise), 'w')
    for line in loca_error_pos_list:
        f.write(str(line) + '\n')
    f.close()

    f = open(os.path.join(saved_path, 'error_x_noise%d.txt' % noise), 'w')
    for line in loca_error_x_list:
        f.write(str(line) + '\n')
    f.close()

    f = open(os.path.join(saved_path, 'error_y_noise%d.txt' % noise), 'w')
    for line in loca_error_y_list:
        f.write(str(line) + '\n')
    f.close()

    f = open(os.path.join(saved_path, 'error_yaw_noise%d.txt' % noise), 'w')
    for line in loca_error_yaw_list:
        f.write(str(line) + '\n')
    f.close()

    f = open(os.path.join(saved_path, 'error_matrix_noise%d.txt' % noise), 'w')
    for line in loca_error_matrix:
        f.write(str(line) + '\n')
    f.close()






    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
