import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .rdsnet_head import RdsnetHead


@HEADS.register_module
class RdsRetinaHead(RdsnetHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 rep_channels=32,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        print('RdsRetinaHead init funcation!')
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RdsRetinaHead, self).__init__(
            num_classes, in_channels, rep_channels=rep_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        print("RdsRetinaHead init_layer funcation!")
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.rep_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.rep_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,   ### 3*3个anchor，80个类别
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.retina_rep = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.rep_channels * 2, 3, padding=1) # 保持不变

    def init_weights(self):
        print("RdsRetinaHead init_weight funcation!")
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.rep_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)
        normal_init(self.retina_rep, std=0.01)

    def forward_single(self, x):
        ## 重写了父类的forward_single函数
        print('RdsRetinaHead forward_single funcation!')
        cls_feat = x
        reg_feat = x
        rep_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        for rep_conv in self.rep_convs:
            rep_feat = rep_conv(rep_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        obj_rep = self.retina_rep(rep_feat)

        # print(len(cls_feat),cls_feat[0].size(),cls_feat[1].size())
        # print(len(bbox_pred),bbox_pred[0].size(),bbox_pred[1].size())
        # print(len(obj_rep),obj_rep[0].size(),obj_rep[1].size())

        return cls_score, bbox_pred, obj_rep
