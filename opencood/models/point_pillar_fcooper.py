# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion


class PointPillarFCooper(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarFCooper, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = SpatialFusion()

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        # localization modify. in_channel should be 128*2*cav_num？
        # self.loc_head = nn.Conv2d(128 * 2*2, 6, kernel_size=1)
        #self.loc_head = nn.Linear(100*352*256*2, 6)

        self.loc_head = nn.Sequential( #(512,100,352)
            # # (start)for smaller voxel size 0.1
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # (512,400,1408)->(512,200,704)
            # nn.Conv2d(512, 512, (3, 3), padding=1),  # (512,200,704)
            # nn.LeakyReLU(0.01),
            # # (end)for smaller voxel size 0.1
            #(start)for smaller voxel size 0.2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # (512,200,704)->(512,100,352)
            nn.Conv2d(512, 512, (3, 3), padding=1),  # (512,100,352)
            nn.LeakyReLU(0.01),
            #(end)for smaller voxel size 0.2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), #(512,50,176)
            nn.Conv2d(512, 512, (3, 3), padding=1), #(512,50,176)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), #(512,25,88)
            nn.Conv2d(512, 512,(3,3), padding=1), #(512,25,88)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), #(512,12,44)
            nn.Conv2d(512, 512, (3,3), padding=1), #(512,12,44)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=(2,4), padding=0), #(512,6,11)
            nn.Conv2d(512, 512, (3,3), padding=1),#(512,6,11)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=(3,2), padding=0),#(512,2,5)
            nn.Conv2d(512, 512, (3,3), padding=1),#(512,2,5)
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=(2,5), padding=0), #(512,1,1)
            nn.Flatten(),
            nn.Linear(512,256, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 3, bias=True)
        )

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        # localization modify 0322
        # add a flag, if convert the features of two agents from A,B to B,A, find the wrong localization while inference
        loca_inference_reverse_flg = data_dict['loca_inference_reverse_flg']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        fused_feature = self.fusion_net(spatial_features_2d, record_len, False, False)


        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        # localization modify
        # in some cases, the agent length is 2, but the processed feature contains only 1, so the tensor dim is 1,
        # then it cannot be processed by localization_fusion_net, give up this kind of data, go to next round training
        # potential reason: the two agents have no overlap points, not sure, need to further debug
        if int(batch_dict['record_len'].sum()) == batch_dict['spatial_features_2d'].shape[0]:
            dim_match_flg = True
        else:
            dim_match_flg = False
            print('feature dim and vehicle quantity not match')

        # locm = self.loc_head(localization_feature.reshape(1, -1))
        if dim_match_flg:
            # localization modify
            # localization_feature = spatial_features_2d.view(1, 100*352*256*2)
            localization_feature = self.fusion_net(spatial_features_2d, record_len, True, loca_inference_reverse_flg)
            locm = self.loc_head(localization_feature)
        else:
            locm = None
        #locm_hidden = nn.functional.relu(self.loc_head_layer1(localization_feature))
        #locm = self.loc_head_layer2(locm_hidden)

        output_dict = {'psm': psm,
                       'rm': rm,
                       #localization modify
                       'locm': locm,
                       'dim_match_flg': dim_match_flg
                       }

        return output_dict
