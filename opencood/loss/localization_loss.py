# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.transformation_utils import transformation_to_x
import math


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None): #localization_loss_method_flg: bool=False):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        #self.localization_loss_method_flg = localization_loss_method_flg
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets


        diff = input - target
        # if self.localization_loss_method_flg:
        #     diff = 0

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class LocalizationLoss(nn.Module):
    def __init__(self, args):
        super(LocalizationLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        self.loss_dict = {}

        # localization modify
        self.loc_weight = args['loc']
        #self.localization_loss_method_flg = True

    def forward(self, output_dict, target_dict #, localization_loss_method_flg=False # , relative_pose_for_loss, gt_relative_pose_for_loss
                ):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict['rm']
        psm = output_dict['psm']
        locm = output_dict['locm']
        targets = target_dict['targets']

        relative_pose_for_loss = target_dict['pose']['relative_pose_for_loss']
        gt_relative_pose_for_loss = target_dict['pose']['gt_relative_pose_for_loss']

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()

        box_cls_labels = target_dict['pos_equal_one']
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                 targets)
        loc_loss_src =\
            self.reg_loss_func(box_preds_sin,
                               reg_targets_sin,
                               weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        # localization modify
        # localization loss
        gt_locm = relative_pose_for_loss-gt_relative_pose_for_loss
        for i in range(gt_locm.size(0)):

            if gt_locm[i, 2] > math.pi:
                gt_locm[i, 2] = gt_locm[i, 2]-2*math.pi
            elif gt_locm[i, 2] < -math.pi:
                gt_locm[i, 2] = gt_locm[i, 2]+2*math.pi
        #localization_loss_src = self.reg_loss_func(locm, gt_locm)#, localization_loss_method_flg = self.localization_loss_method_flg)
        #loc_loss = localization_loss_src.sum() / rm.shape[0]
        localization_loss_src1 = self.reg_loss_func(torch.index_select(locm, 1, torch.tensor([0]).cuda(0)),
                                                    torch.index_select(gt_locm, 1, torch.tensor([0]).cuda(0)))
        localization_loss_src2 = self.reg_loss_func(torch.index_select(locm, 1, torch.tensor([1]).cuda(0)),
                                                    torch.index_select(gt_locm, 1, torch.tensor([1]).cuda(0)))
        localization_loss_src3 = self.reg_loss_func(torch.index_select(locm, 1, torch.tensor([2]).cuda(0)),
                                                    torch.index_select(gt_locm, 1, torch.tensor([2]).cuda(0)))
        localization_loss_src1 = localization_loss_src1.sum() / rm.shape[0]
        localization_loss_src2 = localization_loss_src2.sum() / rm.shape[0]
        localization_loss_src3 = 1.5*57.3 * localization_loss_src3.sum() / rm.shape[0]

        loc_loss = localization_loss_src1 + localization_loss_src2 + localization_loss_src3

        print('locm:', locm)
        print('true:', (relative_pose_for_loss - gt_relative_pose_for_loss))
        print('relative_pose_for_loss', relative_pose_for_loss)
        print('gt_relative_pose_for_loss', gt_relative_pose_for_loss)
        #print('localization_loss_src:', localization_loss_src)
        print('localization_loss_src:', localization_loss_src1, " ", localization_loss_src2, " ", localization_loss_src3)



        #pose_error_for_loss = torch.abs(locm-(relative_pose_for_loss-gt_relative_pose_for_loss))
        #loc_loss = pose_error_for_loss.sum()# * self.loc_weight[1]
            # pose_error_for_loss(1, 2) * self.loc_weight[2] + \
            # pose_error_for_loss(1, 3) * self.loc_weight[3] + \
            # pose_error_for_loss(1, 4) * self.loc_weight[4] + \
            # pose_error_for_loss(1, 5) * self.loc_weight[5] + \
            # pose_error_for_loss(1, 6) * self.loc_weight[6]

        # localization modify (add loc loss)
        # total_loss = reg_loss + conf_loss + loc_loss
        total_loss = loc_loss
        #total_loss = reg_loss + conf_loss

        self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': reg_loss,
                               'conf_loss': conf_loss,
                               'loc_loss': loc_loss,
                               'loc_loss_src1': localization_loss_src1,
                               'loc_loss_src2': localization_loss_src2,
                               'loc_loss_src3': localization_loss_src3
                               })

        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        loc_loss = self.loss_dict['loc_loss']
        loc_loss_src1 = self.loss_dict['loc_loss_src1']
        loc_loss_src2 = self.loss_dict['loc_loss_src2']
        loc_loss_src3 = self.loss_dict['loc_loss_src3']
        if pbar is None:
            '''
            print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                  " || Loc Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), conf_loss.item(), reg_loss.item()))
            '''
            #localization modify:
            print("[epoch %d][%d/%d], || Loss: %.4f || Detection Loss: %.4f"
                  " || Localization Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), conf_loss.item()+reg_loss.item(), loc_loss.item()))
        else:
            '''
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                 " || Loc Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), conf_loss.item(), reg_loss.item()))
            '''
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Detection Loss: %.4f"
                 " || Localization Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), conf_loss.item()+reg_loss.item(), loc_loss.item()))

        writer.add_scalar('Regression_loss', reg_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss', conf_loss.item(),
                          epoch*batch_len + batch_id)
        # localization modify
        writer.add_scalar('Localization_loss', loc_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('loc_loss_src1', loc_loss_src1.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('loc_loss_src2', loc_loss_src2.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('loc_loss_src3', loc_loss_src3.item(),
                          epoch * batch_len + batch_id)

