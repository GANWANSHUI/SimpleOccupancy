# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import sys
sys.path.append('/home/wsgan/project/bev/dgp')
#from dgp.datasets import SynchronizedSceneDataset
import pickle
import torch
import pdb
import cv2
import random, numpy

from .mono_dataset import MonoDataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import sys
sys.path.append("..")


class NuscDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NuscDataset, self).__init__(*args, **kwargs)

        self.split = 'train' if self.is_train else 'val'
        self.data_path = os.path.join(self.opt.dataroot, 'nuscenes')
        self.depth_path = os.path.join(self.opt.dataroot, 'depth_full')
        self.match_path = os.path.join(self.opt.dataroot, 'match')
        self.point_cloud_path = os.path.join(self.opt.dataroot, 'point_cloud_full')
        self.point_cloud_label_path = os.path.join(self.opt.dataroot, 'point_cloud_{}_label/label_52_0.4_surface_fix_num30_depth_52'.format(self.split))

        version = 'v1.0-trainval'

        self.nusc = NuScenes(version=version, dataroot=self.data_path, verbose=False)


        if self.opt.data_type == 'all':
            with open('datasets/nusc/{}.txt'.format(self.split), 'r') as f:
                self.filenames = f.readlines()

        else:
            print('please define data type!!')
            exit()


        self.camera_ids_list = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right',
                           'front', 'front_left', 'back_left', 'back', 'back_right']

        self.camera_names_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
                             'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def get_info(self, inputs, index_temporal, do_flip):
        inputs[("color", 0, -1)] = []

        if self.is_train:
            if self.opt.use_sfm_spatial:
                inputs["match_spatial"] = []

            for idx, i in enumerate(self.frame_idxs[1:]):
                inputs[("color", i, -1)] = []
                inputs[("pose_spatial", i)] = []

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = [] 

            inputs["pose_spatial"] = []
            inputs['depth'] = []

        else:
            inputs[('K_ori', 0)] = [] 
            inputs['depth'] = []

            if self.opt.volume_depth:
                inputs["pose_spatial"] = []


        if self.opt.use_t != 'No' or self.opt.gt_pose:
            for idx, i in enumerate(self.frame_idxs[0:2]):
                inputs[('gt_pose', i)] = []


        inputs['point_cloud'] = []
        inputs['point_cloud_path'] = []
        inputs['point_cloud_label'] = []


        inputs['width_ori'], inputs['height_ori'], inputs['id'] = [], [], []


        rec = self.nusc.get('sample', index_temporal)

        # for point cloud
        cam_sample = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        point_cloud = np.load(os.path.join(self.point_cloud_path, cam_sample['filename'][:-4] + '.npy'))
        inputs['point_cloud'].append(point_cloud[:, :3])

        if self.opt.evl_score and not self.is_train:
            inputs['point_cloud_path'].append(str(os.path.join(self.point_cloud_path, cam_sample['filename'][:-4] + '.npy')))

            # label
            # print(os.path.join(self.point_cloud_path, cam_sample['filename'][:-4] + '.npy'))
            point_cloud_label = np.load(os.path.join(self.point_cloud_label_path, cam_sample['filename'][:-4] + '.npy'), allow_pickle=True)
            point_cloud_label = dict(point_cloud_label.item())
            inputs['point_cloud_label'].append(point_cloud_label)


        for index_spatial in range(6):

            camera_ids = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
            camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

            cam_sample = self.nusc.get('sample_data', rec['data'][camera_names[index_spatial]])
            inputs['id'].append(camera_ids[index_spatial])


            color = self.loader(os.path.join(self.data_path, cam_sample['filename']))
            inputs['width_ori'].append(color.size[0])
            inputs['height_ori'].append(color.size[1])


            depth = np.load(os.path.join(self.depth_path, cam_sample['filename'][:-4] + '.npy'))
            inputs['depth'].append(depth.astype(np.float32))


            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 0, -1)].append(color)

            ego_spatial = self.nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])

            if self.is_train or self.opt.volume_depth:
                pose_0_spatial = Quaternion(ego_spatial['rotation']).transformation_matrix
                '''
                Definition of a particular sensor (lidar/radar/camera) as calibrated on a particular vehicle. 
                All extrinsic parameters are given with respect to the ego vehicle body frame. All camera images come undistorted and rectified.
                '''
                pose_0_spatial[:3, 3] = np.array(ego_spatial['translation'])

                inputs["pose_spatial"].append(pose_0_spatial.astype(np.float32))

            K = np.eye(4).astype(np.float32)
            K[:3, :3] = ego_spatial['camera_intrinsic']
            inputs[('K_ori', 0)].append(K)

            # 以上收集一个相机的： RT K RGB depth
            # load sequence rgb image
            if self.opt.use_t != 'No' and not self.is_train:

                for idx, i in enumerate(self.frame_idxs[1:2]):
                    if i == -1:
                        index_temporal_i = cam_sample['prev']

                    elif i == 1:
                        index_temporal_i = cam_sample['next']

                    cam_sample_i = self.nusc.get('sample_data', index_temporal_i)
                    color_i = self.loader(os.path.join(self.data_path, cam_sample_i['filename']))
                    inputs[("color", i, -1)].append(color_i)


            if self.is_train:

                if self.opt.use_sfm_spatial:
                    pkl_path = os.path.join(os.path.join(self.match_path, cam_sample['filename'][:-4] + '.pkl'))

                    # try:
                    with open(pkl_path, 'rb') as f:
                        match_spatial_pkl = pickle.load(f)
                    inputs['match_spatial'].append(match_spatial_pkl['result'].astype(np.float32))


                for idx, i in enumerate(self.frame_idxs[1:]):
                    if i == -1:
                        index_temporal_i = cam_sample['prev']
                    elif i == 1:
                        index_temporal_i = cam_sample['next']
                    cam_sample_i = self.nusc.get(
                        'sample_data', index_temporal_i)
                    ego_spatial_i = self.nusc.get(
                        'calibrated_sensor', cam_sample_i['calibrated_sensor_token'])

                    K = np.eye(4).astype(np.float32)
                    K[:3, :3] = ego_spatial_i['camera_intrinsic']

                    inputs[('K_ori', i)].append(K)

                    color_i = self.loader(os.path.join(self.data_path, cam_sample_i['filename']))

                    if do_flip:
                        color_i = color_i.transpose(pil.FLIP_LEFT_RIGHT)

                    inputs[("color", i, -1)].append(color_i)




        if self.is_train or self.opt.volume_depth:

            if self.is_train:
                for index_spatial in range(6):
                    for i in [-1, 1]:
                        pose_0_spatial = inputs["pose_spatial"][index_spatial]
                        pose_i_spatial = inputs["pose_spatial"][(index_spatial+i)%6] # 围绕一个转动方向取相邻相机的RT：

                        gt_pose_spatial = np.linalg.inv(pose_i_spatial) @ pose_0_spatial
                        inputs[("pose_spatial", i)].append(gt_pose_spatial.astype(np.float32))


            for idx, i in enumerate(self.frame_idxs):
                if self.is_train:
                    inputs[('K_ori', i)] = np.stack(inputs[('K_ori', i)], axis=0)

                    if i != 0:
                        inputs[("pose_spatial", i)] = np.stack(inputs[("pose_spatial", i)], axis=0)
                        # stack the RT in the same time
                else:
                    inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0)

            if 'depth' in inputs.keys():
                inputs['depth'] = np.stack(inputs['depth'], axis=0)

            inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)

        else:
            inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0) 
            inputs['depth'] = np.stack(inputs['depth'], axis=0)

            if self.opt.volume_depth:
                inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)


        for key in ['width_ori', 'height_ori']:
            inputs[key] = np.stack(inputs[key], axis=0)


def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4)
    inv = torch.cat([inv, bottom_row], 0)
    return inv


def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv
