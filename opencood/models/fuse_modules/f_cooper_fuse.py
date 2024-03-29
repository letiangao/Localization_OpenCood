# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of F-cooper maxout fusing.
"""
import torch
import torch.nn as nn


class SpatialFusion(nn.Module):
    def __init__(self):
        super(SpatialFusion, self).__init__()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, loc_flg, loca_inference_reverse_flg):
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []
        out_split_xx = []

        for xx in split_x:
            if loc_flg:
                #xx = xx.view(1, 2 * 256 * 100 * 352)
                split_xx = torch.tensor_split(xx, 2) # here 2 indicates 2 agent, not batch size 2
                for yy in split_xx:
                    out_split_xx.append(yy)
                # localization modify 0322
                # add a flag, if convert the features of two agents from A,B to B,A, find the wrong localization while inference
                if loca_inference_reverse_flg:
                    out_split_xx = list(reversed(out_split_xx))

                xx = torch.cat(out_split_xx,dim=1)
                #xx = xx.view(1, 2 * 256 * 100 * 352)
            else:
                xx = torch.max(xx, dim=0, keepdim=True)[0]
            out.append(xx)
            out_split_xx = []
        return torch.cat(out, dim=0)