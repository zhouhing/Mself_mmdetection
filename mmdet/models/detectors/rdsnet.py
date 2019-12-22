from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from .. import builder
from mmdet.core import bbox2result, mask2result, multi_apply
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
主要执行入口文件，在此文件中进一步调用各部分网络结构
主干网络，neck网络，bbox网络等组成部分
'''

@DETECTORS.register_module
class RDSNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 bbox_head,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 mbrm_cfg=None,
                 neck=None,
                 pretrained=None):
        print("RDSNet init funcation!")
        print(backbone,'\n',neck,'\n',bbox_head,'\n',train_cfg,'\n',test_cfg)
        print("~~~~~~~~~~~~~~~~~~"*10)
        super(RDSNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.mask_head = builder.build_head(mask_head)
        self.init_extra_weights()
        self.mbrm_cfg = mbrm_cfg
        if mbrm_cfg is not None:
            self.mbrm = MBRM(mbrm_cfg)
        else:
            self.mbrm = None

    def init_extra_weights(self):
        print("RDSNet init_extra_weight funcation!")
        self.mask_head.init_weights()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        print("RDSNet forward_train funcation!")
        print("img:",len(img),img[0].size(),img[1].size())
        # print("img_metas:",img_metas)
        print("gt_bboxes:",len(gt_bboxes),gt_bboxes[0].shape,gt_bboxes[1].shape)
        print("gt_masks:",len(gt_masks),gt_masks[0].shape,gt_masks[1].shape)
        print("gt_bboxes_ignore:",gt_bboxes_ignore)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # print(img[0].size(),img[0].permute(1,2,0).size())
        # print(gt_masks[0].shape)
        # plt.imshow(img[0].cpu().permute(1,2,0).numpy())
        # plt.show()
        # plt.imshow(gt_masks[0][0])
        # plt.show()
        x = self.extract_feat(img)
        print("RDSNet extract_feat:",len(x),len(x[0]),x[0].size(),x[1].size())
        outs = self.bbox_head(x) ##   cls_scores,bbox_preds,obj_reps 目标流，分类信息，位置信息，目标表示信息（2K*d：representation）
        print("RDSNet outs:", len(outs),len(outs[0]))
        # count = 0
        # for i in range(len(outs)):
        #     for j in range(len(outs[0])):
        #         print(outs[i][j].size(), count)
        #         count += 1
        loss_inputs = outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)## 3个目标流的信息+真实标签
        losses, proposals = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # no positive proposals
        if proposals is None:
            return losses


        # needed in  mask generator
        ### TODO stack_shape 在img_meta中添加stack_shape属性
        c, h, w = img.size()[1:4]
        print(img_metas)
        print(img.size())
        for img_meta in img_metas:
            print(img_meta)### 将在batch_size中处理的的图片的统一大小记录下来
            img_meta['stack_shape'] = (h, w, c) # 字典赋值，每张图片对应的标准化处理字典的新变量的增加，congfig中的collect
        print(img_metas[0])
        print(img_metas[1])
        pred_masks = self.mask_head(x, proposals['pos_obj'], img_metas)

        losses_masks, final_masks, final_boxes = self.mask_head.loss(pred_masks, proposals['gt_bbox'],
                                                                     proposals['gt_mask'], img_metas, self.train_cfg)
        losses.update(losses_masks) # 在字典中在添加losses_masks

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        print("RDSNet simple_test funcation!")
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bboxes = [bbox[0] for bbox in bbox_list]
        labels = [bbox[1] for bbox in bbox_list]
        pos_objs = [bbox[2] for bbox in bbox_list]
        # needed in mask generator
        c, h, w = img.size()[1:4]

        ### TODO 在img_meta中添加stack_shape属性
        print("rdsnet simple_test img_meta:",img_meta)
        for meta in img_meta:
            meta['stack_shape'] = (h, w, c) # 将在batch_size中处理的的图片的统一大小记录下来

        pred_masks = self.mask_head(x, pos_objs, img_meta)
        pred_masks = self.mask_head.get_masks(pred_masks, bboxes, img_meta, self.test_cfg, rescale=rescale)

        if self.mbrm is not None:
            bboxes = [self.mbrm.get_boxes(self.mbrm(pred_mask, bbox[:, :-1]), bbox[:, -1:])
                      for pred_mask, bbox in zip(pred_masks, bboxes)]

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in zip(bboxes, labels)
        ]
        mask_results = [
            mask2result(pred_mask, det_labels, self.bbox_head.num_classes, self.test_cfg.mask_thr_binary)# 二值化mask
            for pred_mask, det_labels in zip(pred_masks, labels)
        ]
        return bbox_results[0], mask_results[0]


class MBRM(nn.Module):
    """
    Mask based Boundary Refinement Module.
    """

    def __init__(self, cfg):
        print("MBRM funcation!")
        print("cfg.gamma:",cfg.gamma,"cfg.kernel_size:",cfg.kernel_size)
        super(MBRM, self).__init__()
        self.gamma = cfg.gamma
        self.kernel_size = cfg.kernel_size
        self.weight = nn.Parameter(torch.Tensor(1, 1, cfg.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.weight.data = torch.tensor(
            [[[-0.2968, -0.2892, -0.2635, -3.2545, -0.1874, 2.5041, 0.0196, 0.0651, 0.0917]]])
        self.bias.data = torch.tensor([-1.9536])

    def forward(self, masks, boxes):
        """
        Refine boxes with masks.
        :param masks: Size: [num_dets, h, w]
        :param boxes: Size: [num_dets, 4], absolute coordinates
        :return:
        """
        print("MBRM forward funcation!")
        num_dets, h, w = masks.size()
        if num_dets == 0:
            return None
        gamma = self.gamma

        horizon, _ = masks.max(dim=1)  # 取得每行中最大的值
        vertical, _ = masks.max(dim=2)  # 取得每列中最大的值

        gridx = torch.arange(w, device=masks.device, dtype=masks.dtype).view(1, -1)  # （1，w）
        gridy = torch.arange(h, device=masks.device, dtype=masks.dtype).view(1, -1)  # （1，h）

        sigma_h = ((boxes[:, 2] - boxes[:, 0]) * gamma).view(-1, 1)  ## 扩大boxes的宽，展开成一列
        sigma_h.clamp_(min=1.0)  ### 比1小的值全都置为1
        sigma_v = ((boxes[:, 3] - boxes[:, 1]) * gamma).view(-1, 1)  ## 扩大boxes的高，展开成一列
        sigma_v.clamp_(min=1.0)  ### 比1小的值全都置为1

        sigma_h = 2 * sigma_h.pow(2)
        sigma_v = 2 * sigma_v.pow(2)

        p = int((self.kernel_size - 1) / 2)
        pl = F.conv1d(horizon.view(num_dets, 1, -1), self.weight, self.bias, padding=p).squeeze(1)
        pr = F.conv1d(horizon.view(num_dets, 1, -1), self.weight.flip(dims=(2,)), self.bias, padding=p).squeeze(1)
        pt = F.conv1d(vertical.view(num_dets, 1, -1), self.weight, self.bias, padding=p).squeeze(1)
        pb = F.conv1d(vertical.view(num_dets, 1, -1), self.weight.flip(dims=(2,)), self.bias, padding=p).squeeze(1)

        lweight = torch.exp(-(gridx - boxes[:, 0:1]).float().pow(2) / sigma_h)
        rweight = torch.exp(-(gridx - boxes[:, 2:3]).float().pow(2) / sigma_h)
        tweight = torch.exp(-(gridy - boxes[:, 1:2]).float().pow(2) / sigma_v)
        bweight = torch.exp(-(gridy - boxes[:, 3:4]).float().pow(2) / sigma_v)

        lweight = torch.where(lweight > 0.0044, lweight, lweight.new_zeros(1)) # 大于0.0044取lweight，否则取1
        rweight = torch.where(rweight > 0.0044, rweight, rweight.new_zeros(1))
        tweight = torch.where(tweight > 0.0044, tweight, tweight.new_zeros(1))
        bweight = torch.where(bweight > 0.0044, bweight, bweight.new_zeros(1))

        activate_func = torch.sigmoid

        pl = activate_func(pl) * lweight
        pr = activate_func(pr) * rweight
        pt = activate_func(pt) * tweight
        pb = activate_func(pb) * bweight

        dl = pl / pl.sum(dim=1, keepdim=True) # 最后一步傅里叶变换，获得概率
        dr = pr / pr.sum(dim=1, keepdim=True)
        dt = pt / pt.sum(dim=1, keepdim=True)
        db = pb / pb.sum(dim=1, keepdim=True)

        return [dl, dt, dr, db]

    def get_boxes(self, boundary_distribution, labels):
        print("MBRM get_boxes funcation!")
        if boundary_distribution is None:
            return labels.new_zeros(0, 5)
        (d_l, d_t, d_r, d_b) = boundary_distribution
        l = torch.argmax(d_l, dim=1).float()
        r = torch.argmax(d_r, dim=1).float()
        t = torch.argmax(d_t, dim=1).float()
        b = torch.argmax(d_b, dim=1).float()
        boxes = torch.stack([l, t, r, b], dim=1)

        return torch.cat([boxes, labels], -1)
