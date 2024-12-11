import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

from mmdet.models import DETECTORS
from mmdet3d.models.detectors import MVXFasterRCNN
from .cam_stream_lss import LiftSplatShoot
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

@DETECTORS.register_module()
class BEVF_FasterRCNN(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, lss=False, lc_fusion=False, camera_stream=False,
                camera_depth_range=[4.0, 45.0, 1.0], img_depth_loss_weight=1.0,  img_depth_loss_method='kld',
                grid=0.6, num_views=6, se=False,
                final_dim=(900, 1600), pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, imc=256, lic=384, **kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.

        """
        super(BEVF_FasterRCNN, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        self.se = se
        self.pc_range = pc_range
        if camera_stream:
            self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64, 
            pc_range=pc_range, final_dim=final_dim, downsample=downsample)
        if lc_fusion:
            if se:
                self.seblock = SE_Block(lic)
            self.reduc_conv = ConvModule(
                lic + imc,
                lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
            
        self.freeze_img = kwargs.get('freeze_img', False)
        self.init_weights(pretrained=kwargs.get('pretrained', None))
        self.freeze()

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.lift:
                for param in self.lift_splat_shot_vis.parameters():
                    param.requires_grad = False


    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_size = img.size() 
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        if self.lift:
            BN, C, H, W = img_feats[0].shape
            batch_size = BN // self.num_views
            img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W)
            rots = []
            trans = []
            rots_depth = []
            trans_depth = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                rot_depth_list = []
                trans_depth_list = []
                for mat in img_metas[sample_idx]['lidar2img']:
                    mat = torch.Tensor(mat)
                    rot_list.append(mat.inverse()[:3, :3].to(img_feats_view.device))
                    trans_list.append(mat.inverse()[:3, 3].view(-1).to(img_feats_view.device))
                    rot_depth_list.append(mat[:3, :3].to(img_feats_view.device))
                    trans_depth_list.append(mat[:3, 3].view(-1).to(img_feats_view.device))
                rot_list = torch.stack(rot_list, dim=0)
                trans_list = torch.stack(trans_list, dim=0)
                rot_depth_list = torch.stack(rot_depth_list, dim=0)
                trans_depth_list = torch.stack(trans_depth_list, dim=0)
                rots.append(rot_list)
                trans.append(trans_list)
                rots_depth.append(rot_depth_list)
                trans_depth.append(trans_depth_list)
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            rots_depth = torch.stack(rots_depth)  # depth transform 4 6 3 3
            trans_depth = torch.stack(trans_depth)  # depth transform

            # 创建缩放矩阵
            scale_x = H / img_size[3]
            scale_y = W / img_size[4]
            scale_matrix = torch.tensor([
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]
            ]).cuda()
            rots_depth_scaled = torch.matmul(scale_matrix, rots_depth)
            trans_depth_scaled = trans_depth * torch.tensor([scale_x, scale_y, 1]).cuda()
            lidar2img_rt = img_metas[sample_idx]['lidar2img']  #### extrinsic parameters for multi-view images

            batch_size = len(points)
            dx = int((self.camera_depth_range[1] - self.camera_depth_range[0]) / self.camera_depth_range[2])
            depth = torch.zeros(batch_size, img_size[1], 1, dx, H, W).cuda() # 创建大小
            depth_mask = torch.zeros(batch_size, img_size[1], 1, H, W).cuda() # 创建大小
            for b in range(batch_size):
                cur_coords = points[b].float()[:, :3]  #取点的xyz
                # lidar2image
                cur_coords = rots_depth_scaled[b].matmul(cur_coords.transpose(1, 0))
                cur_coords += trans_depth_scaled[b].reshape(-1, 3, 1)

                # get 2d coords
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-4, 1e4)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
                # imgaug

                cur_coords = cur_coords[:, :2, :].transpose(1, 2)
                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = (
                    (cur_coords[..., 0] < H)
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < W)
                    & (cur_coords[..., 1] >= 0)
                )

                for c in range(on_img.shape[0]):
                    masked_coords = cur_coords[c, on_img[c]].long()  # 点云投影到图像坐标
                    masked_dist = dist[c, on_img[c]]  # 对应深度

                    # 使用字典存储每个坐标的深度值
                    coord_dict = {}
                    for i in range(masked_coords.shape[0]):
                        coord = (masked_coords[i, 0].item(), masked_coords[i, 1].item())
                        if coord not in coord_dict:
                            coord_dict[coord] = []
                        coord_dict[coord].append(masked_dist[i].item())

                    # 计算每个坐标的深度均值并更新 depth 和 depth_mask
                    for coord, img_depths in coord_dict.items():
                        depth_mask[b, c, 0, coord[0], coord[1]] = 1  # 有效的像素点
                        for img_depth in img_depths:
                            if img_depth < self.camera_depth_range[0] or img_depth > self.camera_depth_range[1]:
                                continue
                            depth[b, c, 0, int((img_depth-self.camera_depth_range[0]) / self.camera_depth_range[2]), coord[0], coord[1]] += 1 / len(img_depths)  # 稀疏的深度约束图（用于计算loss）
            #TODO, 已完成depth 作为离散分布真值的计算  depth：torch.size(B, N, D, H, W)
            img_bev_feat, depth_dist = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas)

            # print(img_bev_feat.shape, pts_feats[-1].shape)
            if pts_feats is None:
                pts_feats = [img_bev_feat] ####cam stream only
            else:
                if self.lc_fusion:
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear', align_corners=True)
                    pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
        return dict(
            img_feats = img_feats,
            pts_feats = pts_feats,
            depth_dist = depth_dist
        )
        # return (img_feats, pts_feats, depth_dist)
    
    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                # pts_feats, img_feats, img_metas, rescale=rescale)
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            if img_depth is not None:
                loss_depth = self.depth_dist_loss(depth_dist, img_depth, loss_method=self.img_depth_loss_method, img=img) * self.img_depth_loss_weight
                losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses
    
    def depth_dist_loss(self, predict_depth_dist, gt_depth, loss_method='kld', img=None):
        # predict_depth_dist: B, N, D, H, W
        # gt_depth: B, N, H', W'
        B, N, D, H, W = predict_depth_dist.shape
        guassian_depth, min_depth = gt_depth[..., 1:], gt_depth[..., 0]
        mask = (min_depth>=self.camera_depth_range[0]) & (min_depth<=self.camera_depth_range[1])
        mask = mask.view(-1)
        guassian_depth = guassian_depth.view(-1, D)[mask]
        predict_depth_dist = predict_depth_dist.permute(0, 1, 3, 4, 2).reshape(-1, D)[mask]
        if loss_method=='kld':
            loss = F.kl_div(torch.log(predict_depth_dist), guassian_depth, reduction='mean', log_target=False)
        elif loss_method=='mse':
            loss = F.mse_loss(predict_depth_dist, guassian_depth)
        else:
            raise NotImplementedError
        return loss

