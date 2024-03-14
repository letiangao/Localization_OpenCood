# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


from opencood.utils.common_utils import torch_tensor_to_numpy
import math
import statistics
def cal_localization_error(loca_error_matrix,loca_error_x_list,loca_error_y_list,loca_error_yaw_list,loca_error_pos_list, pose_gt, pose_initial, pose_error_model):
    """
    pose_gt: tensor, relative pose ground truth, x,y,yaw
    pose_initial: tensor, relative pose with added error, x,y,yaw
    pose_error_model: model output pose, it's relative pose error, x error, y error, yaw error
    """
    #convert tensor to numpy:
    pose_gt_np = torch_tensor_to_numpy(pose_gt[0]) #common_utils.py
    pose_initial_np = torch_tensor_to_numpy(pose_initial[0])
    pose_model_np = pose_initial_np-torch_tensor_to_numpy(pose_error_model[0])

    error = pose_gt_np - pose_model_np
    print('errors initial:', pose_gt_np-pose_initial_np)
    print('errors from model:', error)
    error_x = abs(error[0])
    error_y = abs(error[1])
    error_yaw = abs(error[2])
    error_pos = math.sqrt(error[0]**2 + error[1]**2)
    loca_error_matrix.append(error)
    loca_error_x_list.append(error_x)
    loca_error_y_list.append(error_y)
    loca_error_yaw_list.append(error_yaw)
    loca_error_pos_list.append(error_pos)
    loc_error_x_mean = statistics.mean(loca_error_x_list)
    loc_error_y_mean = statistics.mean(loca_error_y_list)
    loc_error_yaw_mean = statistics.mean(loca_error_yaw_list)
    loc_error_mean = statistics.mean(loca_error_pos_list)

    return loca_error_matrix,loc_error_x_mean,loc_error_y_mean,loc_error_yaw_mean,loc_error_mean

