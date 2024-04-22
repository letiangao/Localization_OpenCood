# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import gc
import os
import statistics

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

from collections import OrderedDict # localization modify
from opencood.visualization import vis_utils # localization modify
import open3d as o3d # localization modify
import time
from opencood.visualization import vis_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    visual_localization = False #True #localization modify
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    #(start)localization dataset visualization
    vis_subflag = True
    if visual_localization:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs = []
        for _ in range(50):
            vis_aabbs.append(o3d.geometry.LineSet())
    # (end)localization dataset visualization



    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):

            # localization modify: if the scenario contains only ego CAV, which means the other CAV is out of range, skip this scenario
            if batch_data['ego']['record_len'].sum().item()< 2*hypes['train_params']['batch_size']: # localization modify
                continue  # localization modify
            # the model will be evaluation mode during validation
            aaa= batch_data['ego']['feature_num']
            del batch_data['ego']['merged_feature_dict']
            print("distance:",batch_data['ego']['distance'])
            del batch_data['ego']['distance']

            print("i:", i)
            batch_data['ego']['loca_inference_reverse_flg'] = False #this para only used in inference
            # (start)localization dataset visualization
            if visual_localization:
                pcd, aabbs = \
                    vis_utils.visualize_single_sample_dataloader(batch_data['ego'],
                                                                 vis_pcd,
                                                                 'hwl',
                                                                 mode='intensity')
                print(pcd)
                if vis_subflag == True: #i == 0:
                    vis_subflag = False
                    vis.add_geometry(pcd)
                    for ii in range(len(vis_aabbs)):
                        index = ii if ii < len(aabbs) else -1
                        vis_aabbs[ii] = vis_utils.lineset_assign(vis_aabbs[ii], aabbs[index])
                        vis.add_geometry(vis_aabbs[ii])

                for ii in range(len(vis_aabbs)):
                    index = ii if ii < len(aabbs) else -1
                    vis_aabbs[ii] = vis_utils.lineset_assign(vis_aabbs[ii], aabbs[index])
                    vis.update_geometry(vis_aabbs[ii])

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)
            # (end)localization dataset visualization

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            batch_data['ego']['label_dict']['pose'] = OrderedDict() # localization modify
            batch_data['ego']['label_dict']['pose'].update({
                                   'relative_pose_for_loss': batch_data['ego']['relative_pose_for_loss'],
                                   'gt_relative_pose_for_loss': batch_data['ego']['gt_relative_pose_for_loss']}) # localization modify

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            a = batch_data['ego']['record_len']


            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                if not ouput_dict['dim_match_flg']:
                    continue
                del ouput_dict['dim_match_flg']
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                #localization modify: third and fourth arguments are for localization
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'],
                                        #localization_loss_method_flg = True
                                       )
                                       #batch_data['ego']['relative_pose_for_loss'],
                                       #batch_data['ego']['gt_relative_pose_for_loss']
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    if not ouput_dict['dim_match_flg']:
                        continue
                    del ouput_dict['dim_match_flg']
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'],
                                           #localization_loss_method_flg= True
                                           #batch_data['ego']['relative_pose_for_loss'],
                                           #batch_data['ego']['gt_relative_pose_for_loss']
                                           )


            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

            if i > 5000: #localization modify
                break

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    # localization modify: if the scenario contains only ego CAV, which means the other CAV is out of range, skip this scenario
                    if batch_data['ego']['record_len'].sum().item()<2*hypes['train_params']['batch_size']:  # localization modify
                        continue  # localization modify
                    del batch_data['ego']['merged_feature_dict']
                    batch_data['ego']['loca_inference_reverse_flg'] = False  # this para only used in inference

                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)

                    batch_data['ego']['label_dict']['pose'] = OrderedDict()  # localization modify
                    batch_data['ego']['label_dict']['pose'].update({
                        'relative_pose_for_loss': batch_data['ego']['relative_pose_for_loss'],
                        'gt_relative_pose_for_loss': batch_data['ego'][
                            'gt_relative_pose_for_loss']})  # localization modify

                    ouput_dict = model(batch_data['ego'])
                    if not ouput_dict['dim_match_flg']:
                        continue
                    del ouput_dict['dim_match_flg']

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
                    if i>1000: #1500: #localization modify
                        break
            # save validate loss details to txt
            f = open(os.path.join(saved_path, 'validation_loss_at_epoch%d.txt' % epoch), 'w')
            for line in valid_ave_loss:
                f.write(str(line)+'\n')
            f.close()

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    #(start) visualizaton localization
    if visual_localization:
        vis.destroy_window()

if __name__ == '__main__':
    main()
