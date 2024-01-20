# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import time

import cv2
import numpy as np

from opencood.utils import box_utils
from opencood.utils import common_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x_to_world
from opencood.utils.transformation_utils import transformation_to_x
import copy
from opencood.utils.common_utils import torch_tensor_to_numpy

def transLidarToPcd(lidar_ego, lidar_cav, pose_gt, pose_initial, pose_error_model,cav_lidar_range):
    # lidar_ego: np array, original lidar from ego vehicle, not projected
    # lidar_cav: np array, original lidar from cav, not projected
    # pose_gt: tensor, relative pose ground truth, x,y,yaw
    # pose_initial: tensor, relative pose with added error, x,y,yaw
    # pose_error_model: model output pose, it's relative pose error, x error, y error, yaw error
    projected_lidar_stack = []
    #convert tensor to numpy:
    pose_gt_np = torch_tensor_to_numpy(pose_gt[0]) #common_utils.py
    pose_initial_np = torch_tensor_to_numpy(pose_initial[0])
    pose_model_np = pose_initial_np-torch_tensor_to_numpy(pose_error_model[0])

    # lidar process:
    # filter lidar
    lidar_ego = shuffle_points(lidar_ego)
    lidar_cav = shuffle_points(lidar_cav)
    # remove points that hit itself
    lidar_ego = mask_ego_points(lidar_ego)
    lidar_cav = mask_ego_points(lidar_cav)

    # transformation calculation:
    # def x_to_world(pose):pose:x,y,z,roll,yaw,pitch in deg
    transformation_gt = x_to_world([pose_gt_np[0], pose_gt_np[1], 0, 0, np.degrees(pose_gt_np[2]), 0])
    transformation_initial = x_to_world([pose_initial_np[0], pose_initial_np[1], 0, 0, np.degrees(pose_initial_np[2]), 0])
    transformation_model = x_to_world([pose_model_np[0], pose_model_np[1], 0, 0, np.degrees(pose_model_np[2]), 0])



    # project the lidar to ego space


    lidar_ego[:, 3] = 0.9  # set color for localization visualization
    #lidar_np_gt[:, 3] = 0.9  # set color for localization visualization
    lidar_ego = mask_points_by_range(lidar_ego,
                                     cav_lidar_range) #self.params['preprocess']['cav_lidar_range']

    lidar_cav_gt = copy.deepcopy(lidar_cav)
    lidar_cav_gt[:, 3] = 0.2  # set color for localization visualization
    lidar_cav_gt[:, :3] = \
        box_utils.project_points_by_matrix_torch(lidar_cav_gt[:, :3],
                                                 transformation_gt)
    lidar_cav_gt = mask_points_by_range(lidar_cav_gt,
                                     cav_lidar_range)

    lidar_cav_initial = copy.deepcopy(lidar_cav)
    lidar_cav_initial[:, 3] = 0.95  # set color for localization visualization
    lidar_cav_initial[:, :3] = \
        box_utils.project_points_by_matrix_torch(lidar_cav_initial[:, :3],
                                                 transformation_initial)
    lidar_cav_initial = mask_points_by_range(lidar_cav_initial,
                                        cav_lidar_range)


    lidar_cav_model = copy.deepcopy(lidar_cav)
    lidar_cav_model[:, 3] = 0.75  # set color for localization visualization
    lidar_cav_model[:, :3] = \
        box_utils.project_points_by_matrix_torch(lidar_cav_model[:, :3],
                                                 transformation_model)
    lidar_cav_model = mask_points_by_range(lidar_cav_model,
                                        cav_lidar_range)


    #lidar_cav[:, 3] = 0.5  # set color for localization visualization
    #lidar_np_gt[:, 3] = 5  # set color for localization visualization
    # lidar_cav[:, :3] = \
    #         box_utils.project_points_by_matrix_torch(lidar_cav[:, :3],
    #                                                  transformation_matrix)
    projected_lidar_stack.append(lidar_ego) #orange
    projected_lidar_stack.append(lidar_cav_gt) #blue
    projected_lidar_stack.append(lidar_cav_initial) #yellow
    projected_lidar_stack.append(lidar_cav_model) #purple
    pcd = np.vstack(projected_lidar_stack)
    return pcd


