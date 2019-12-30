import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob

INF = 1e8


@HEADS.register_module
class MyFCOSHead(nn.Module):
    # print("class MyFcos!"*5)
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),  ## 被配置文件中的strides修改为（8, 16, 32, 64, 128）
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(MyFCOSHead, self).__init__()
        # print("MyFCOSHead __init__ funcation!")
        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()
        ###  顶层高语义注意力 ##
        self._attention()

    def _attention(self):
        print('attention funcation!')
        self.attention = nn.ModuleList()
        for j in range(len(self.strides) - 1):
            layer = nn.ModuleList()
            for i in range(len(self.strides) - j - 1):
                layer.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.feat_channels,
                                       kernel_size=3, stride=2,padding=1),
                    nn.ReLU(),
                ))
            self.attention.append(layer)

    def _init_layers(self):
        # print("MyFCOSHead _init_layers funcation!")
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
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
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        # print("MyFCOSHead init_weights funcation!")
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        print("MyFCOSHead forward funcation!")
        for i in range(len(feats)):
            print(feats[i].size())
        print('~~~~~~~~~~~~~')
        top_feat=feats[-1]
        for i, atten_conv in enumerate(self.attention):
            print(i, len(atten_conv))
            feature = feats[-1]
            for j in range(len(atten_conv)):
                feature=atten_conv[j](feature)
                print("feature.size():",feature.size())
            print(feats[i].size())
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        # print("MyFCOSHead forward_simgle funcation!")
        cls_feat = x
        reg_feat = x
        # print(cls_feat.size())
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        # max_cls_score, index = cls_score.max(dim=1)
        # print(len(max_cls_score),max_cls_score.size(),len(index),index.size())
        # print("cls_score.size():",cls_score.size(),"bbox_pred:",bbox_pred.size(),'centerness:',centerness.size())
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        # print('MyFCOSHead loss funcation!' )
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        # print(bbox_preds[0].size())
        # print("cls_scores:",np.array(cls_scores).size,"bboxs_preds:",np.array(bbox_preds).size,"centernesses:",np.array(centernesses).size,
        #       np.array(gt_bboxes).shape,np.array(gt_labels).shape,'img_metas:',np.array(img_metas),'cfg:',cfg,'************\n'*2)

        ############## 5层金字塔的特征图大小  ########################
        # for i in range(len(cls_scores)):
        #     print((cls_scores[i].cpu().detach().numpy()).shape)
        ###########################################################

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        # print("all_level_points:",all_level_points)

        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes,
                                                gt_labels)  # 得到每张图片对应的每层的label，和 bbox_target

        # print("labels:","bbox_targets:")
        # for i in range(len(labels)):
        #     print(labels[i].size(),bbox_targets[i].size())
        # print("!!!!!!!!!!!!!!!!!!!!!!!!")
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness

        # print('变换是狗相同：',((cls_scores[0].reshape(-1,self.cls_out_channels))==(cls_scores[0].permute(0,2,3,1).reshape(-1,self.cls_out_channels))).size())
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]  ## 展开为，每个像素点对应于80个类别的概率值
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]  ## 展开为，每个像素点的预测边框的位置
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]  ## 展开为，每个像素点对应的的中心度

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)  ## 将各特征层上对应像素点的坐标进行综合，得到原图所有像素点的坐标
        flatten_bbox_targets = torch.cat(bbox_targets)
        ########################################################
        ########################################################
        # ########################################################
        # for i in range(len(cls_scores)):
        #     print('###'*10)
        #     print((cls_scores[i].cpu().detach().numpy()).shape)
        #########################################################
        # print("cls_scores.size():",len(cls_scores))
        # print("Flatten_cls_scores.size():",flatten_cls_scores.size())
        # print("bbox_preds:",len(bbox_preds))
        # print("bbox_preds:",flatten_bbox_preds.size())
        # print("centernesses:",len(centernesses))
        # print("centernesses:",flatten_centerness.size())
        # print("num_img:",num_imgs)
        # repeat points to align with bbox_preds
        # print("all_level_points:",all_level_points[-1].size(),len(all_level_points),num_imgs)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in
             all_level_points])  ##重复两次生成两张图片的点（由FPN的5个不同尺寸的特征图决定5个不同的H*W相加得到总点数），cat到一起
        # print(all_level_points[-1].repeat(num_imgs,1).size(),all_level_points[-1].size())
        # print("flatten_all_level_points:", flatten_points.size(), len(flatten_points))
        pos_inds = flatten_labels.nonzero().reshape(-1)  # 获取label中不为0的相应坐标
        num_pos = len(pos_inds)  # 一共有多少不为0的label
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0  ## 使用Focal loss进行类别loss的计算
        # print("loss_cls:",loss_cls,loss_cls.shape)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]  # 像素点的label不为0的，位置
        pos_centerness = flatten_centerness[pos_inds]  # 像素点的label不为0的中心度

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]  # 像素点label不为0的，target真实bbox
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)  # 对应真实bbox的中心度
            pos_points = flatten_points[pos_inds]

            ##########  为了对下面进行IOU_Loss 计算做准备   ####################
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            ################## 计算IOU_loss  ###################
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())  # 计算IOU_loss
            # print("loss_bbox:",loss_bbox,loss_bbox.shape)
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
            # print("centernesses:",loss_centerness,loss_centerness.size())
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        # print('MyFCOSHeadget_point funcation!' )
        mlvl_points = []
        # print("self.strides:", self.strides)
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        # print('MyFCOSHead get_point_single!'  )
        # print("featmap_size:", featmap_size, "strides:", stride)
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)  ### stride is (8, 16, 32, 64, 128)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        # print("x_range:",x_range,"y_range:",y_range)
        y, x = torch.meshgrid(y_range, x_range)
        # print("x_:", x, '\nx.reshape(-1):',x.reshape(-1),"\ny_:", y,"\ny.reshape(-1):",y.reshape(-1))
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        # print("points:", points.size())
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        # print('MyFCOSHead fcos_target function!' )
        assert len(points) == len(self.regress_ranges)  # 5
        # print(type(points[0]), type(gt_bboxes_list[0]), type(gt_labels_list[0]))

        # print("fcos_target:", np.array(points).shape,points[0].size(), np.array(gt_bboxes_list).shape,gt_bboxes_list[0].size(), np.array(gt_labels_list).shape,gt_labels_list[0].size())

        num_levels = len(points)  # 5
        # expand regress ranges to align with points
        # print("self.regress_ranges:", self.regress_ranges)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)  # 将回归限制范围扩展到各层对应的各个点
        ]

        # print("the number of points:")
        # num_points = 0
        # for i in range(len(points)):
        #     num_points += points[i].size(0)
        #     print(points[i].size())
        #     print(points[i].new_tensor(self.regress_ranges[i]).size())
        #     print(points[i].new_tensor(self.regress_ranges[i]).expand_as(points[i]).size())
        # print("num_points:", num_points)
        # for i in range(len(expanded_regress_ranges)):
        #     print("expanded_regress_ranges{}:".format(i), expanded_regress_ranges[i].size())
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)  # 匹配所有点的个数，将回归限制范围拼接到同一行中
        # print("concat_regress_ranges:", concat_regress_ranges.size())
        concat_points = torch.cat(points, dim=0)  ## 将所有点拼接到同一行中
        # print("fcos_points:", concat_points.size())
        # get labels and bbox_targets of each image
        # print("gt_bboxes_list:",gt_bboxes_list,np.array(gt_bboxes_list).shape,"\ngt_labels_list:",gt_labels_list,np.array(gt_labels_list).shape)
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]  ## 每层点的个数
        # print("num_points:",num_points)
        # print("labels_list1:",(labels_list),labels_list[0].size(),labels_list[1].size(),np.array(labels_list).shape)

        labels_list = [labels.split(num_points, 0) for labels in labels_list]  # # num_points是一个数组，按数组划分为5组labels标签
        # print("labels_list2:", len(labels_list), np.array(labels_list).shape)
        # print("bbox_targets_list1:", len(bbox_targets_list), bbox_targets_list[0].size())
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list  ### 同上
        ]
        # print("bbox_targets_list2:", len(bbox_targets_list), np.array(bbox_targets_list).shape,)

        # concat per level image
        ## 将label和bbox_target 对应到各个特征层的位置
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        # print("concat_label:",len(concat_lvl_labels),concat_lvl_labels[0].size())
        # print("concat_lvl_bbox_targets:",len(concat_lvl_bbox_targets),concat_lvl_bbox_targets[0].size())
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        # print('MyFCOSHead fcos_target_single funcation!' )
        # print("fcos_target_single:", gt_bboxes, gt_labels, points, regress_ranges)
        num_points = points.size(0)  # 5层特征图上所有像素点的总的数目
        num_gts = gt_labels.size(0)  # 原图上有多少个实例对象
        # print("num_points:",num_points,"num_gts:",num_gts)
        if num_gts == 0:  # 没有实例则全为0
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)

        # print(areas)

        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)  ##  - *sizes (torch.Size ot int...)-沿第一维（即纵列）重复的次数,匹配每个点的面积

        # print(areas)

        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)  # 每个点对应于每个实例的范围，2是限制范围（-1(0)，64）(64,128)
        # print("regress_ranges:",regress_ranges,regress_ranges.size())

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)  # 对应于每个点，匹配像素点和真实框，准备由此范围过滤某些负样本点
        # print("gt_bboxes:",gt_bboxes)
        # print("points:",points,points.size())
        xs, ys = points[:, 0], points[:, 1]  # 获得点的 横 纵坐标（X，Y）

        # print("xs:",xs,xs.size(),xs[:,None],"\nys:",ys,ys.size(),ys[:,None])
        xs = xs[:, None].expand(num_points, num_gts)  ##  - sizes(torch.Size or int...)-需要扩展的大小
        ys = ys[:, None].expand(num_points, num_gts)  ## [:,None]先变为2维向量，变成共有num_point行，1 列的数据
        # print("xs:", xs,xs.size(), "\nys:", ys,ys.size())

        # print("gt_bboxes[...,0]:",gt_bboxes[...,0])
        left = xs - gt_bboxes[..., 0]
        # print("gt_bboxes[...,2]:", gt_bboxes[..., 2])
        right = gt_bboxes[..., 2] - xs
        # print("gt_bboxes[...,1]:", gt_bboxes[..., 1])
        top = ys - gt_bboxes[..., 1]
        # print("gt_bboxes[...,3]:", gt_bboxes[..., 3])
        bottom = gt_bboxes[..., 3] - ys

        bbox_targets = torch.stack((left, top, right, bottom), -1)
        # print("bbox_targets:",bbox_targets,bbox_targets.size(),bbox_targets.min(-1),bbox_targets.min(-1)[0].size())

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0  # 像素点点在实例框内,将不在实例框内的边框标记为False
        # print("inside_gt_bbox_mask：",inside_gt_bbox_mask,inside_gt_bbox_mask.size())
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                                       max_regress_distance >= regress_ranges[..., 0]) & (
                                       max_regress_distance <= regress_ranges[..., 1])  # 获得在每层限制范围内的像素点
        # print("inside_regress_range：",inside_regress_range,inside_regress_range.size())
        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF  # 将不在实例框内的边框面积标记为 INF　无穷大
        # print('areas of inside_gt_bboxes_mask:',areas)
        areas[inside_regress_range == 0] = INF  # 将超过各层限制的预测边框，标记为 INF，，上下两个综合得到最最终满足条件的边框
        ## 标记两个无穷大是为了后面选取面积最小值做准备
        # print('areas of inside_regress_range:', areas,areas.size())
        # print("areas.min(dim=1):",areas.min(dim=1))
        min_area, min_area_inds = areas.min(dim=1)  # 当这个像素点生成的框属于多个类别时，将这个框分配给生成框面积小的实例，
        # min_area 代表当前生成所有类别的框中面积最小的面积
        # min_area_inds代表的是在该个像素点的位置生成的对应几个类别的框中，选取的面积最小的实例边框的下标（在所有类别中的位置下标）
        # print("min_area:",min_area,min_area.size(),"min_area_inds:",min_area_inds,min_area_inds.size(),type(min_area_inds))
        # print("gt_labels:",gt_labels,gt_labels.size())
        labels = gt_labels[min_area_inds]  # 获取对应最小面积的实例目标类别标签，最小目标的标签
        # print("labels:",labels,labels.size())
        labels[min_area == INF] = 0  # 将类别中最小的面积中无穷大的值置为0
        # print("labels:",labels,labels.size())
        bbox_targets = bbox_targets[range(num_points), min_area_inds]  # 获取num_points个边框，在第二维按照面积小的下标，获取相应边框
        # print("bbox_targets:",bbox_targets,bbox_targets.size())

        return labels, bbox_targets  ## 得到生成框对应的类别标签，和框的位置

    def centerness_target(self, pos_bbox_targets):
        # print('MyFCOSHead centerness funcation!')
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
                                     left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                                     top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    ##################################################################################################################
    ################################# 以下属于test阶段使用函数 ##########################################################
    ################################# 以下属于test阶段使用函数 ##########################################################
    ################################# 以下属于test阶段使用函数 ##########################################################
    ##################################################################################################################
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None):
        # print('MyFCOSHead get_bboxes funcation!')
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []

        # print("img_metas:",img_metas,len(img_metas))

        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        # print('MyFCOSHead get_bboxes_single funcation!' )
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)  # 按中心度和类别得分取前几名得分box
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels
