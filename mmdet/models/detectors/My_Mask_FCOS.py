# -*-coding utf-8 -*-
# @Time :2019/12/22 20:09
# @Author : 50317
# To become better
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from .. import builder
from mmdet.core import bbox2result, mask2result, multi_apply
import torch
import torch.nn as nn
import torch.nn.functional as F

@DETECTORS.register_module
class My_Mask_FCOS(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_head=None,
                 mbrm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(My_Mask_FCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained)
        self.mask_head=builder.build_neck(mask_head)
        self.init_extra_weights()
        self.mbrm_cfg=mbrm_cfg
        if self.mbrm_cfg is not None:
            self.mbrm=MBRM(mbrm_cfg)
        else:
            self.mbrm=None


    def init_extra_weights(self):
        print("RDSNet init_extra_weight funcation!")
        self.mask_head.init_weights()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None):
        x=self.extract_feat(img)
        outs=self.bbox_head(x)## cls_score, bbox_pred, centerness
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)  ## 3个目标流的信息+真实标签
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        obj_rep=outs[0]## 将cls_score作为像素关于实例的预测表示


class MBRM(mbrm_cfg=None):
    pass