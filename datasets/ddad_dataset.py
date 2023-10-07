# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import pickle
import cv2


from .mono_dataset import MonoDataset


class DDADDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(DDADDataset, self).__init__(*args, **kwargs)

        self.split = 'train' if self.is_train else 'val'
        self.dataroot = self.opt.dataroot

        self.rgb_path = os.path.join(self.opt.dataroot, 'raw_data')
        self.depth_path = os.path.join(self.opt.dataroot, 'depth')
        self.mask_path = os.path.join(self.opt.dataroot, 'mask')
        self.match_path = os.path.join(self.opt.dataroot, 'match')
        self.point_cloud_path = os.path.join(self.opt.dataroot, 'point_cloud2023')
        self.point_cloud_label_path = os.path.join(self.opt.dataroot, 'label/point_cloud_{}_label_52_0.0_center_g_fix_num30_new'.format(self.split))


        with open('datasets/ddad/{}.txt'.format(self.split), 'r') as f:
            self.filenames = f.readlines()

        with open('datasets/ddad/info_{}.pkl'.format(self.split), 'rb') as f:

            self.info = pickle.load(f)

        self.camera_ids = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
        self.camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_07', 'CAMERA_09', 'CAMERA_08', 'CAMERA_06']

    
    def get_info(self, inputs, index_temporal, do_flip):
        inputs[("color", 0, -1)] = []
        inputs[("pose_spatial")] = []

        if self.is_train:
            if self.opt.use_sfm_spatial:
                inputs["match_spatial"] = []

            for idx, i in enumerate(self.frame_idxs[1:]):
                inputs[("color", i, -1)] = []
                inputs[("pose_spatial", i)] = []

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = [] 
            
            inputs['mask_ori'] = []
            inputs['depth'] = []
            inputs['point_cloud'] = []
            inputs['point_cloud_path'] = []

        else:
            inputs[('K_ori', 0)] = [] 
            inputs['depth'] = []

            if self.opt.evl_score:
                inputs['point_cloud'] = []
                inputs['point_cloud_path'] = []
                inputs['point_cloud_label'] = []


        inputs['width_ori'], inputs['height_ori'], inputs['id'] = [], [], []

        scene_id = self.info[index_temporal]['scene_name']


        # for train label
        if self.is_train:
            point_cloud = np.load(os.path.join(self.point_cloud_path, scene_id, 'LIDAR', index_temporal + '.npy'))
            inputs['point_cloud'].append(point_cloud)
            inputs['point_cloud_path'].append(str(os.path.join(self.point_cloud_path, scene_id, 'LIDAR', index_temporal + '.npy')))


        if self.opt.evl_score and not self.is_train:

            point_cloud = np.load(os.path.join(self.point_cloud_path, scene_id, 'LIDAR', index_temporal + '.npy'))
            inputs['point_cloud'].append(point_cloud)
            inputs['point_cloud_path'].append(str(os.path.join(self.point_cloud_path, scene_id, 'LIDAR', index_temporal + '.npy')))

            # label
            point_cloud_label = np.load(os.path.join(self.point_cloud_label_path, scene_id, 'LIDAR', index_temporal + '.npy'),  allow_pickle=True)
            point_cloud_label = dict(point_cloud_label.item())
            inputs['point_cloud_label'].append(point_cloud_label)


        for index_spatial in range(6):
            inputs['id'].append(self.camera_ids[index_spatial])
            color = self.loader(os.path.join(self.rgb_path, scene_id, 'rgb', 
                                self.camera_names[index_spatial], index_temporal + '.png'))
            inputs['width_ori'].append(color.size[0])
            inputs['height_ori'].append(color.size[1])
            
        
            if not self.is_train:
                depth = np.load(os.path.join(self.depth_path, scene_id, 'depth',
                            self.camera_names[index_spatial], index_temporal + '.npy'))
                inputs['depth'].append(depth.astype(np.float32))

            else:
                depth = np.load(os.path.join(self.depth_path + '_train_new', scene_id, 'depth',
                                             self.camera_names[index_spatial], index_temporal + '.npy'))
                inputs['depth'].append(depth.astype(np.float32))


            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 0, -1)].append(color)

            # if self.is_train or self.opt.volume_depth:

            pose_0_spatial = self.info[index_temporal][self.camera_names[index_spatial]]['extrinsics']['quat'].transformation_matrix
            pose_0_spatial[:3, 3] = self.info[index_temporal][self.camera_names[index_spatial]]['extrinsics']['tvec']

            inputs["pose_spatial"].append(pose_0_spatial.astype(np.float32))

    
            K = np.eye(4).astype(np.float32)
            K[:3, :3] = self.info[index_temporal][self.camera_names[index_spatial]]['intrinsics']
            inputs[('K_ori', 0)].append(K)

            if self.is_train:

                mask = cv2.imread(os.path.join(self.mask_path, self.camera_names[index_spatial], scene_id, 'mask.png'))
                inputs["mask_ori"].append(mask)

                if self.opt.use_sfm_spatial:
                    pkl_path = os.path.join(self.match_path, scene_id, 'match',
                                            self.camera_names[index_spatial], index_temporal + '.pkl')
                    with open(pkl_path, 'rb') as f:
                        match_spatial_pkl = pickle.load(f)
                    inputs['match_spatial'].append(match_spatial_pkl['result'].astype(np.float32))

                for idx, i in enumerate(self.frame_idxs[1:]): # [0, -1, 1]
                    index_temporal_i = self.info[index_temporal]['context'][idx] # idx: 0 (前一帧),1 (后一帧)

                    K = np.eye(4).astype(np.float32)
                    K[:3, :3] = self.info[index_temporal_i][self.camera_names[index_spatial]]['intrinsics']
                    inputs[('K_ori', i)].append(K)

                    color = self.loader(os.path.join(self.rgb_path, scene_id, 'rgb', 
                                    self.camera_names[index_spatial], index_temporal_i + '.png'))
                    
                    if do_flip:
                        color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
                    inputs[("color", i, -1)].append(color)
                    pose_i_spatial = self.info[index_temporal][self.camera_names[(index_spatial+i)%6]]['extrinsics']['quat'].transformation_matrix
                    pose_i_spatial[:3, 3] = self.info[index_temporal][self.camera_names[(index_spatial+i)%6]]['extrinsics']['tvec']
                    gt_pose_spatial = np.linalg.inv(pose_i_spatial) @ pose_0_spatial
                    inputs[("pose_spatial", i)].append(gt_pose_spatial.astype(np.float32))


            if self.opt.use_t != 'No' and not self.is_train:
                for idx, i in enumerate(self.frame_idxs_t[1:2]):
                    index_temporal_i = self.info[index_temporal]['context'][idx]  # idx: 0 (前一帧),1 (后一帧)
                    if index_temporal_i == -1:
                        color = self.loader(os.path.join(self.rgb_path, scene_id, 'rgb',
                                                         self.camera_names[index_spatial], index_temporal + '.png'))
                        inputs[("color", i, -1)].append(color)  # i: -1 前一帧， 1 后一帧

                    else:
                        color = self.loader(os.path.join(self.rgb_path, scene_id, 'rgb',
                                                         self.camera_names[index_spatial], index_temporal_i + '.png'))
                        inputs[("color", i, -1)].append(color)

        if self.is_train:
            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = np.stack(inputs[('K_ori', i)], axis=0)
                if i != 0:
                    inputs[("pose_spatial", i)] = np.stack(inputs[("pose_spatial", i)], axis=0)

            inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)

            if 'depth' in inputs.keys():
                inputs['depth'] = np.stack(inputs['depth'], axis=0)

        else:
            inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0) 
            inputs['depth'] = np.stack(inputs['depth'], axis=0)
            inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)


        for key in ['width_ori', 'height_ori']:
            inputs[key] = np.stack(inputs[key], axis=0)   





