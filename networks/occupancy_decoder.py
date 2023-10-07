# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import pdb
from collections import OrderedDict

import torch
from ._3DCNN import S3DCNN
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


class VolumeDecoder(nn.Module):

    def __init__(self, opt):
        super(VolumeDecoder, self).__init__()

        self.opt = opt
        self.batch = self.opt.batch_size // 6

        self.near = self.opt.min_depth
        self.far = self.opt.max_depth

        self.register_buffer('xyz_min', torch.from_numpy(
            np.array([self.opt.real_size[0], self.opt.real_size[2], self.opt.real_size[4]])))  # -80, -80, 0
        self.register_buffer('xyz_max', torch.from_numpy(
            np.array([self.opt.real_size[1], self.opt.real_size[3], self.opt.real_size[5]])))  # 80, 80, 8

        self.ZMAX = self.opt.real_size[1]

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


        length_pose_encoding = 3


        if self.opt.position == 'embedding':
            input_channel = self.opt.input_channel
            self.pos_embedding = torch.nn.Parameter(torch.ones(
                [1, input_channel, self.opt.voxels_size[1], self.opt.voxels_size[2], self.opt.voxels_size[0]]))

        elif self.opt.position == 'embedding1':
            input_channel = self.opt.input_channel
            xyz_in_channels = 1 + 3

            embedding_width = 192
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
        scene_centroid_y = 0.0
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


        activate_fun = nn.ReLU(inplace=True)
        if self.opt.aggregation == '3dcnn':
            out_put = self.opt.out_channel
            self._3DCNN = S3DCNN(input_planes=input_channel, out_planes=out_put, planes=self.opt.con_channel,
                                 activate_fun=activate_fun, opt=opt)
        else:
            print('please define the aggregation')
            exit()


    def feature2vox_simple(self, features, pix_T_cams, cam0_T_camXs, __p, __u):

        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)

        _, C, Hf, Wf = features.shape

        sy = Hf / float(self.opt.height)
        sx = Wf / float(self.opt.width)

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_

        feat_mems_ = self.vox_util.unproject_image_to_mem(
            features,
            basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, self.Z, self.Y, self.X)

        # feat_mems_ shape： torch.Size([6, 128, 200, 8, 200])
        feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X # torch.Size([1, 6, 128, 200, 8, 200])

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X
        feat_mem = feat_mem.permute(0, 1, 4, 3, 2) # [0, ...].unsqueeze(0) # ZYX -> XYZ

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
            rng += torch.rand_like(rng[:, [0]])

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
            probs[:, -1] = 1
            density = torch.sigmoid(density)
            probs[~mask_outbbox] = density

            # accumulate
            probs = probs.cumsum(dim=1).clamp(max=1)
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


    def forward(self, features, inputs, outputs = {}, is_train=True):

        __p = lambda x: basic.pack_seqdim(x, self.batch)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.batch)

        self.outputs = outputs

        # 2D to 3D
        Voxel_feat= self.feature2vox_simple(features[0][:6], inputs[('K', 0, 0)], inputs['pose_spatial'], __p, __u)

        # position prior
        if self.opt.position == 'embedding':
            Voxel_feat = Voxel_feat * self.pos_embedding

        elif self.opt.position == 'embedding1':
            if is_train:
                embedding = self.embeddingnet(self.meta_data)
                embedding = torch.reshape(embedding, [self.opt.B, self.Z, self.Y, self.X, self.opt.input_channel]).permute(0, 4, 3, 2, 1)
                self.pos_embedding_save.data = embedding

            else:
                embedding = self.pos_embedding_save

            if self.opt.position == 'embedding1':
                Voxel_feat = Voxel_feat * embedding

            else:
                print('please define the opt.position')
                exit()

        elif self.opt.position == 'No':
            pass

        else:
            print('please define the opt.position')
            exit()


        # 3D aggregation
        Voxel_feat_list = self._3DCNN(Voxel_feat)

        # rendering
        rays_o = __u(inputs['rays_o', 0])
        rays_d = __u(inputs['rays_d', 0])

        if is_train:
            for scale in self.opt.scales:
                depth, rgb_marched = self.get_density(rays_o, rays_d, Voxel_feat_list[scale], is_train, inputs)
                self.outputs[("disp", scale)] = depth.reshape(self.opt.cam_N, self.opt.render_h, self.opt.render_w).unsqueeze(1).clamp(self.opt.min_depth, self.opt.max_depth)

        else:
            depth, rgb_marched = self.get_density(rays_o, rays_d, Voxel_feat_list[0], is_train, inputs)
            self.outputs[("disp", 0)] = depth.reshape(self.opt.cam_N, self.opt.render_h,
                                                      self.opt.render_w).unsqueeze(1).clamp(self.opt.min_depth, self.opt.max_depth)

            if self.opt.evl_score:
                voxel = Voxel_feat_list[0]
                self.outputs['True_center'] = None
                self.outputs['empty'] = None

                if self.opt.val_depth:
                    depth_empty = inputs['val_depth_empty'].float()
                    self.outputs['depth_empty'] = self.get_scores(depth_empty, voxel)

        self.outputs[("density")] = Voxel_feat_list[0]

        return self.outputs
