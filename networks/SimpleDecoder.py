# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import pdb
from collections import OrderedDict

import torch

import torch
from ._3DCNN import S3DCNN, S3DCNN_multi_scale
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time
from sys import path
path.append("/home/wsgan/project/bev/SimpleOccupancy/utils")
import geom
import vox
import basic
import render

path.append('/home/wsgan/project/bev/S3DO/networks/view_transform/query')
from surroundocc_query import surroundocc_query



class SimpleDecoder(nn.Module):

    def __init__(self, opt):
        super(SimpleDecoder, self).__init__()

        self.opt = opt
        self.batch = self.opt.batch_size // 6

        self.near = self.opt.min_depth
        self.far = self.opt.max_depth

        self.register_buffer('xyz_min', torch.from_numpy(np.array([self.opt.real_size[0], self.opt.real_size[2], self.opt.real_size[4]])))  # -80, -80, 0
        self.register_buffer('xyz_max', torch.from_numpy(np.array([self.opt.real_size[1], self.opt.real_size[3], self.opt.real_size[5]])))  # 80, 80, 8

        self.ZMAX = self.opt.real_size[1]

        # pdb.set_trace()

        self.Z = self.opt.voxels_size[0]  # 16
        self.Y = self.opt.voxels_size[1]  # 256
        self.X = self.opt.voxels_size[2]  # 256


        self.Z_final = self.Z
        self.Y_final = self.Y
        self.X_final = self.X


        self.stepsize = self.opt.stepsize
        self.num_voxels = self.Z_final * self.Y_final * self.X_final  # 200*200*8
        N_samples = int(np.linalg.norm(np.array([self.Z_final // 2, self.Y_final // 2, self.X_final // 2]) + 1) / self.stepsize) + 1

        self.stepsize_log = self.stepsize
        self.interval = self.stepsize

        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)  # +-80, 8 # 0.7528
        self.register_buffer('rng', torch.arange(N_samples)[None].float())

        if self.opt.use_t != 'No' and self.opt.position != 'embedding1' and self.opt.position != 'embedding_t':
            self.opt.input_channel = self.opt.input_channel * 2

        length_pose_encoding = 3


        if self.opt.position == 'embedding':
            input_channel = self.opt.input_channel
            self.pos_embedding = torch.nn.Parameter(torch.ones(
                [1, input_channel, self.opt.voxels_size[1], self.opt.voxels_size[2], self.opt.voxels_size[0]]))

        elif self.opt.position == 'embedding1':
            input_channel = self.opt.input_channel
            xyz_in_channels = 1 + 3 # + 1 * 2 * length_pose_encoding # 3 + 1

            embedding_width = 192  # 128
            embedding_depth = 5

            self.embeddingnet = nn.Sequential(
                nn.Linear(xyz_in_channels, embedding_width), nn.ReLU(inplace=True),
                *[nn.Sequential(nn.Linear(embedding_width, embedding_width), nn.ReLU(inplace=True))
                    for _ in range(embedding_depth - 2)], nn.Linear(embedding_width, self.opt.input_channel),)

            nn.init.constant_(self.embeddingnet[-1].bias, 0)
            self.pos_embedding1 = None
            self.pos_embedding_save = torch.nn.Parameter(torch.zeros([1, input_channel, self.opt.voxels_size[1], self.opt.voxels_size[2], self.opt.voxels_size[0]]), requires_grad= False)

        else:
            self.pos_embedding = None
            self.pos_embedding1 = None
            input_channel = self.opt.input_channel

        scene_centroid_x = 0.0
        scene_centroid_y = 0.0  # default 1.0, 正方向朝下
        scene_centroid_z = 0.0

        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])

        self.register_buffer('scene_centroid', torch.from_numpy(scene_centroid).float())

        self.bounds = (self.opt.real_size[0], self.opt.real_size[1],
                       self.opt.real_size[2], self.opt.real_size[3],
                       self.opt.real_size[4], self.opt.real_size[5])
        #  bounds = (-52, 52, -52, 52, 0, 6)


        self.vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds, position = self.opt.position, length_pose_encoding = length_pose_encoding, opt = self.opt,
            assert_cube=False)

        if self.opt.position != 'No' and self.opt.position != 'embedding':
            self.meta_data = self.vox_util.get_meta_data(cam_center=torch.Tensor([[1.2475, 0.0673, 1.5356]]), camB_T_camA=None).to('cuda')

        self.act_shift = 0

        if self.opt.trans2voxel == 'interpolation':
            activate_fun = nn.ReLU(inplace=True)
            if self.opt.aggregation == '3dcnn':

                # if self.opt.use_semantic:

                #     self.opt.out_channel = 17

                # else:
                #     self.opt.out_channel = 1

                # pdb.set_trace()
                self._3DCNN = S3DCNN_multi_scale(input_planes=input_channel, out_planes=self.opt.out_channel, planes=self.opt.con_channel,
                                     activate_fun=activate_fun, opt=opt)

            else:
                print('please define the aggregation')
                exit()



        if self.opt.view_trans == 'query':

            # single
            _dim_ = [64]  # feature channel for different down scale feature size
            num_cams = 6

            _ffn_dim_ = [256, 512, 1024]

            volume_h_ = [200, 50, 25]
            volume_w_ = [200, 50, 25]
            volume_z_ = [16, 4, 2]

            # volume_h_ = [256, 50, 25]
            # volume_w_ = [256, 50, 25]
            # volume_z_ = [16, 4, 2]

            # volume_h_ = [self.opt.voxels_size[2], 50, 25]
            # volume_w_ = [self.opt.voxels_size[1], 50, 25]
            # volume_z_ = [self.opt.voxels_size[0], 4, 2]

            # self.opt.voxels_size[0] = 16  # 16
            # self.opt.voxels_size[1] = 200  # 256
            # self.opt.voxels_size[2] = 200  # 256

            _num_points_ = [2, 4, 8]
            _num_layers_ = [1, 3, 6]
            img_channels = [64, 512, 512]
            # point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]

            range_depth = self.opt.max_depth

            point_cloud_range = [-range_depth, -range_depth, 0, range_depth, range_depth, 6.0]

            occ_size = [200, 200, 16]

            num_classes = 1

            upsample_strides = [1, 2, 1, 2, 1, 2, 1]

            transformer_template = dict(
                type='PerceptionTransformer',
                embed_dims=_dim_,
                num_cams=num_cams,

                encoder=dict(
                    type='OccEncoder',
                    num_layers=_num_layers_,
                    pc_range=point_cloud_range,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='OccLayer',
                        attn_cfgs=[
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=point_cloud_range,
                                num_cams=num_cams,

                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=_dim_,
                                    num_points=_num_points_,
                                    num_levels=1),
                                embed_dims=_dim_, )],

                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        embed_dims=_dim_,
                        conv_num=2,

                        operation_order=('cross_attn', 'norm',
                                         'ffn', 'norm', 'conv')
                        # operation_order=('cross_attn',)
                    )))

            # pdb.set_trace()
            self.occdecoder = surroundocc_query(num_classes=num_classes, embed_dims=_dim_,
                                                volume_h=volume_h_,
                                                volume_w=volume_w_,
                                                volume_z=volume_z_,
                                                out_indices=[0, 2, 4, 6], upsample_strides=upsample_strides,
                                                img_channels=img_channels, transformer_template=transformer_template)



    def feature2vox_simple(self, features, pix_T_cams, cam0_T_camXs, __p, __u):

        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)

        # pdb.set_trace()
        _, C, Hf, Wf = features.shape

        sy = Hf / float(self.opt.height)
        sx = Wf / float(self.opt.width)

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_


        # pdb.set_trace()
        feat_mems_ = self.vox_util.unproject_image_to_mem(features,
            basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, self.Z, self.Y, self.X)


        # pdb.set_trace()
        # feat_mems_ shape： torch.Size([6, 128, 200, 8, 200])
        feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X # torch.Size([1, 6, 128, 200, 8, 200])

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X
        feat_mem = feat_mem.permute(0, 1, 4, 3, 2) # [0, ...].unsqueeze(0) # ZYX -> XYZ

        # pdb.set_trace()

        return feat_mem

    def grid_sampler(self, xyz, *grids, align_corners=True):
        '''Wrapper for the interp operation'''
        # pdb.set_trace()
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1 # XYZ
        grid = grids[0] # BCXYZ # torch.Size([1, 1, 256, 256, 16])
        ret_lst = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1]).squeeze()

        return ret_lst

    def sample_ray(self, rays_o, rays_d, is_train):
        '''Sample query points on rays'''
        rng = self.rng
        if is_train == 'train':
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])  # add some noise to the sample

        step = self.stepsize_log * self.voxel_size * rng
        Zval = step
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * Zval[..., None] #
        rays_pts_depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)

        mask_outbbox = ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)

        return rays_pts, mask_outbbox, Zval, rays_pts_depth

    def activate_density(self, density, interval):
        return 1 - torch.exp(-F.softplus(density) * interval)

    def get_density(self, rays_o, rays_d, Voxel_feat, is_train, inputs):

        eps_time = time.time()
        with torch.no_grad():
            rays_o_i = rays_o[0, ...].flatten(0, 2)  # HXWX3
            rays_d_i = rays_d[0, ...].flatten(0, 2)  # HXWX3
            rays_pts, mask_outbbox, interval, rays_pts_depth = self.sample_ray(rays_o_i, rays_d_i, is_train=is_train)

        mask_rays_pts = rays_pts[~mask_outbbox]
        density = self.grid_sampler(mask_rays_pts, Voxel_feat) # 256,256,16

        if self.opt.render_type == 'prob':

            probs = torch.zeros_like(rays_pts[..., 0])
            probs[:, -1] = 1  # 最后一个点density设置为 1
            density = torch.sigmoid(density)
            probs[~mask_outbbox] = density

            # accumulate
            probs = probs.cumsum(dim=1).clamp(max=1)  # 累加并限制最大值为1
            probs = probs.diff(dim=1, prepend=torch.zeros((rays_pts.shape[:1])).unsqueeze(1).to('cuda'))
            depth = (probs * interval).sum(-1)
            rgb_marched = 0

            # center
            if self.opt.l1_voxel != 'No' and is_train:
                True_center = inputs['True_center'].float()
                self.outputs[("density_center", 0)] = self.get_scores(True_center, Voxel_feat)

                all_empty = inputs['all_empty'].float()
                self.outputs[("all_empty", 0)] = self.get_scores(all_empty, Voxel_feat)

            else:
                pass

        elif self.opt.render_type == 'density':

            alpha = torch.zeros_like(rays_pts[..., 0])
            interval_list = interval[..., 1:] - interval[..., :-1]
            alpha[~mask_outbbox] = self.activate_density(density, interval_list[0, -1])

            weights, alphainv_cum = render.get_ray_marching_ray(alpha)
            depth = (weights * interval).sum(-1)
            rgb_marched = 0


        else:
            print('please define render_type')
            pass

        return depth, rgb_marched

    def get_scores(self, pts, Voxel_feat):
        density = self.grid_sampler(pts, Voxel_feat)
        scores = torch.sigmoid(density)
        return scores



    def get_ego2pix_rt(self, features, pix_T_cams, cam0_T_camXs):

        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)

        # pdb.set_trace()
        # _, C, Hf, Wf = features.shape[-2], features.shape[-1]
        Hf, Wf = features.shape[-2], features.shape[-1]

        sy = Hf / float(self.opt.height)
        sx = Wf / float(self.opt.width)

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_
        ego2featpix = basic.matmul2(featpix_T_cams_, camXs_T_cam0_)


        return ego2featpix, camXs_T_cam0_, Hf, Wf

    def get_voxel(self, features, inputs):

        __p = lambda x: basic.pack_seqdim(x, self.batch)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.batch)

        meta_similarity = None
        meta_feature = None
        curcar2precar = None

        feature_size = self.opt.input_channel

        if self.opt.view_trans == 'simple':

            Voxel_feat = self.feature2vox_simple(features[0][:6], inputs[('K', 0, 0)], inputs['pose_spatial'], __p, __u)


        elif self.opt.view_trans == 'query':
            ego2featpix, camXs_T_cam0_, Hf, Wf = self.get_ego2pix_rt(features[0], inputs[('K', 0, 0)], inputs[
                    'pose_spatial'])  # for simpleocc is ego2img actually

            img_meta = {}
            # ego2featpix = torch.cat([ego2featpix, ego2featpix], dim=0)
            # features = [torch.cat([features[0], features[0]], dim=1)]

            # 以下的两个参数是相互对应的，用query的代码中，用于判断投影之后在相平面的点
            # pdb.set_trace()
            img_meta['lidar2img'] = ego2featpix.to('cpu').numpy()  # numpy
            img_meta['img_shape'] = [(Hf, Wf)]

            img_metas = [img_meta]

            if len(features[0].shape) == 4:
                features[0] = features[0].unsqueeze(0)

            # pdb.set_trace()
            Voxel_feat = self.occdecoder(mlvl_feats=features, img_metas=img_metas)

            # pdb.set_trace()
            Voxel_feat = Voxel_feat[0]# .squeeze(0)
            # [1, 64, 200, 200, 16]

        return Voxel_feat, meta_similarity, meta_feature, curcar2precar, feature_size

    def forward(self, features, inputs, outputs = {}, is_train=True):

        __p = lambda x: basic.pack_seqdim(x, self.batch)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.batch)

        self.outputs = outputs

        Voxel_feat, meta_similarity, meta_feature, curcar2precar, feature_size = self.get_voxel(features, inputs)

        # pdb.set_trace()
        Voxel_feat_list = self._3DCNN(Voxel_feat)
        # pdb.set_trace()

        return Voxel_feat_list

        # rendering
        rays_o = __u(inputs['rays_o', 0])
        rays_d = __u(inputs['rays_d', 0])

        if is_train:
            for scale in self.opt.scales:
                # pdb.set_trace()
                depth, rgb_marched = self.get_density(rays_o, rays_d, Voxel_feat_list[scale], is_train, inputs)
                # depth = self.get_density(inputs, Voxel_feat_list[scale])
                self.outputs[("disp", scale)] = depth.reshape(self.opt.cam_N, self.opt.render_h, self.opt.render_w).unsqueeze(1).clamp(self.opt.min_depth, self.opt.max_depth)

        else:
            depth, rgb_marched = self.get_density(rays_o, rays_d, Voxel_feat_list[0], is_train, inputs)
            # depth = self.get_density(inputs, Voxel_feat_list[0])

            self.outputs[("disp", 0)] = depth.reshape(self.opt.cam_N, self.opt.render_h,
                                                      self.opt.render_w).unsqueeze(1).clamp(self.opt.min_depth, self.opt.max_depth)

            # pdb.set_trace()
            if self.opt.evl_score:
                voxel = Voxel_feat_list[0]

                if self.opt.ground_prior:
                    # pdb.set_trace()
                    voxel[..., 0] = voxel[..., 0].max()

                self.outputs['True_center'] = None
                self.outputs['empty'] = None
                self.outputs['True_voxel_group'] = None

                if self.opt.val_depth:
                    depth_empty = inputs['val_depth_empty'].float()
                    self.outputs['depth_empty'] = self.get_scores(depth_empty, voxel)

        self.outputs[("density")] = Voxel_feat_list[0]
        self.outputs[("act_shift")] = self.act_shift

        return self.outputs
