# -*-coding utf-8 -*-
# @Time :2019/12/12 18:21
# @Author : 50317
# To become better
from ..registry import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module
class MyFCOS(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(MyFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)