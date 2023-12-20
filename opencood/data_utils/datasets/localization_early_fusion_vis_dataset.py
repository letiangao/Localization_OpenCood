# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
This is a dataset for early fusion visualization only.
"""
import random
from collections import OrderedDict

import numpy as np
import torch
import copy
from torch.utils.data import DataLoader

from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class LocalizationEarlyFusionVisDataset(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True):
        super(LocalizationEarlyFusionVisDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)
        #(start)for visualization localization:
        # localization modify
        # modify the scenario_database for localization, if contains multiple cavs,
        # divide the scenario into multiple scenarios, each contains only 2 cavs
        modified_scenario_database = OrderedDict()
        scenario_num = 0
        modified_len_record = list()
        for scenario_id, scenario_value in self.scenario_database.items():
            if len(scenario_value) < 2:
                continue
            else:
                cav_id_list = [int(x) for x in scenario_value.keys()]
                cav_id_list.sort()
                for ego_id in cav_id_list[:-1]:
                    for cav_id in [i for i in cav_id_list if i > ego_id]:
                        modified_scenario_database[scenario_num] = OrderedDict()
                        ##modified_scenario_database[scenario_num].update(OrderedDict({str(ego_id): self.scenario_database[scenario_id][str(ego_id)], str(cav_id):self.scenario_database[scenario_id][str(cav_id)]}))
                        temp1 = copy.deepcopy(self.scenario_database[scenario_id][str(ego_id)])
                        temp2 = copy.deepcopy(self.scenario_database[scenario_id][str(cav_id)])
                        modified_scenario_database[scenario_num].update(
                            {str(ego_id): temp1,
                             str(cav_id): temp2})
                        modified_scenario_database[scenario_num][str(ego_id)].update({'ego': True})
                        modified_scenario_database[scenario_num][str(cav_id)].update({'ego': False})
                        # the len_record needs to be modified based on the modified scenario database
                        if not modified_len_record:
                            modified_len_record.append(self.len_record[0])
                        elif scenario_id == 0:
                            pre_len = modified_len_record[-1]
                            modified_len_record.append(
                                pre_len + self.len_record[0])
                        else:
                            pre_len = modified_len_record[-1]
                            modified_len_record.append(
                                pre_len + (self.len_record[scenario_id] - self.len_record[scenario_id - 1]))
                        scenario_num = scenario_num + 1
        self.scenario_database = modified_scenario_database
        self.len_record = modified_len_record
        # (end) for visualization localization

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)
            # all these lidar and object coordinates are projected to ego
            # already.
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            #add lidar that transformed using transformation gt:
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar_gt'])

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)

        # data augmentation
        projected_lidar_stack, object_bbx_center, mask = \
            self.augment(projected_lidar_stack, object_bbx_center, mask)

        # we do lidar filtering in the stacked lidar
        # for localization: watch out the following filter, may cause error
        # when visualize pcd using transformation and gt_transformation,
        # the filter makes the pcd using the transformation one disappear
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'])
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid = \
            box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'],
                                                     self.params['postprocess'][
                                                         'order']
                                                     )
        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'origin_lidar': projected_lidar_stack
             })

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose)

        transformation_matrix = selected_cav_base['params']['transformation_matrix']
        #aaa = transformation_matrix-transformation_matrix1

        gt_transformation_matrix = selected_cav_base['params']['gt_transformation_matrix']
        gt_transformation_matrix[0, 3] = gt_transformation_matrix[0, 3]
        gt_transformation_matrix[1, 3] = gt_transformation_matrix[1, 3]
        #transformation_matrix = selected_cav_base['params']['gt_transformation_matrix']
        bbb = [transformation_matrix[0, 3] - gt_transformation_matrix[0, 3],transformation_matrix[1, 3] - gt_transformation_matrix[1, 3]]
        print(bbb)
        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        temp=copy.deepcopy(lidar_np)
        lidar_np_gt = temp
        lidar_np_gt[:, :3] = \
           box_utils.project_points_by_matrix_torch(lidar_np_gt[:, :3],
                                                    gt_transformation_matrix)
        # project the lidar to ego space
        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)
        #ccc=lidar_np_gt-lidar_np
        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'projected_lidar_gt': lidar_np_gt
             })

        return selected_cav_processed

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask})

        origin_lidar = \
            np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
        origin_lidar = torch.from_numpy(origin_lidar)
        output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict
