from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import time
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms
import pdb
import open3d as o3d

# all_pose, pts_xyz, val_reso, max_height, max_depth, center_label, val_depth


def get_mask(pts_xyz, val_reso = 0.4, max_height = 6, max_depth = 52):

    # mask out the height (0.4, 6)
    mask1 = pts_xyz[..., 2] < val_reso
    mask2 = pts_xyz[..., 2] >= max_height

    xy_range = max_depth

    # x
    mask3 = pts_xyz[..., 0] > xy_range
    mask4 = pts_xyz[..., 0] < -xy_range

    # y
    mask5 = pts_xyz[..., 1] > xy_range
    mask6 = pts_xyz[..., 1] < -xy_range

    # mask out the point cloud close to car, especially for nuscenes
    # x
    mask7 = (pts_xyz[..., 0] < 3.5) & (pts_xyz[..., 0] > -0.5)
    # y
    mask8 = (pts_xyz[..., 1] < 1.0) & (pts_xyz[..., 1] > -1.0)

    mask9 = mask7 & mask8

    mask = mask1 + mask2 + mask3 + mask4 + mask5 + mask6 + mask9

    return mask


def get_occupancy_label_test_group(all_pose, pts_xyz, val_reso, max_height, max_depth, center_label, val_depth):

    # all_pose, pts_xyz = inputs['pose_spatial'], inputs['point_cloud'][0]

    np.random.seed(0)

    # voxel_size = self.opt.max_depth/self.opt.voxels_size[0]  # 乘于2，再除于2，相消
    voxel_size = val_reso
    stride = voxel_size  # empty stride
    # pdb.set_trace()
    # pts_xyz = pts_xyz[0]

    mask = get_mask(pts_xyz, val_reso, max_height, max_depth)


    GT_point = pts_xyz[~mask]
    pts_xyz = GT_point

    # point cloud down sample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz)
    pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
    pcd_down_sample_point = np.asarray(pcd.points)

    # point_for_empty = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    # total_vertex = []
    total_center = []
    total_voxel_group = []  # we define a voxel's 8 vertex and its center as a group

    # http://www.open3d.org/docs/0.8.0/python_api/open3d.geometry.VoxelGrid.html
    # get all camera center
    all_cam_center = np.array([all_pose[:, 0, 3].mean(), all_pose[:, 1, 3].mean(), all_pose[:, 2, 3].mean()])




    # 进行判断的点先进行降采样，其中降采样为voxel分辨率的1/4
    # get vertex and center point
    for i in range(pcd_down_sample_point.shape[0]):
        # get vertex
        vertex = np.asarray(voxel_grid.get_voxel_bounding_points(voxel_grid.get_voxel(pcd_down_sample_point[i, :])))

        # get voxel center
        center_point = voxel_grid.get_voxel_center_coordinate(voxel_grid.get_voxel(pcd_down_sample_point[i, :]))

        instance_group = np.concatenate((vertex, center_point[np.newaxis, :]), axis=0)

        # 需要注意有无量化的误差，导致误判
        total_voxel_group.append(instance_group)
        total_center.append(center_point)

    # 转numpy, 然后去除重复的voxel
    total_voxel_group = np.array(total_voxel_group)
    # pdb.set_trace()
    total_voxel_group = np.unique(total_voxel_group, axis=0)

    # empty 的采样点需要从点云出发，以降低误差， 不适合从center point 出发
    # pcd_down_sample_point = total_center
    if center_label:
        total_center = np.array(total_center)
        total_center = np.unique(total_center, axis=0)
        pcd_down_sample_point = total_center
    else:
        total_center = np.array(pcd_down_sample_point)
        pcd_down_sample_point = total_center
        # print('total center shape:', total_center.shape)


    # total_vertex = np.array(total_vertex)
    # total_vertex = np.unique(total_vertex, axis=0)
    # inputs['True_vertex'] = torch.from_numpy(total_vertex)

    inputs['True_center'] = torch.from_numpy(total_center)
    inputs['True_voxel_group'] = torch.from_numpy(total_voxel_group)

    # 如果调整成距离表达则无这个影响
    if inputs['True_center'].shape[0] < 100:
        pass

    else:
        # get empty point
        total_empty_point = []
        sample = 'fix_step'

        if sample == 'fix_step':
            for i in range(len(pcd_down_sample_point)):
                vector = pcd_down_sample_point[i] - all_cam_center
                length = np.linalg.norm(vector)
                norm_vector = vector / length

                # pdb.set_trace() 这里可以
                for j in range(1, int((length // stride) - (1.5 // stride))):
                    # for j in range(1, min(4, int((length // stride) - (1.5 // stride)))):

                    sampled_point = pcd_down_sample_point[i] - stride * j * norm_vector
                    sampled_point = sampled_point.tolist()
                    # if sampled_point not in total_empty_point:
                    total_empty_point.append(sampled_point)

            total_empty_point = np.array(total_empty_point)
            total_empty_point = torch.from_numpy(total_empty_point)


        elif sample == 'fix_num':

            # fix number
            N_step = 50
            pts_xyz = torch.from_numpy(pcd_down_sample_point).float()
            vector = pts_xyz - all_cam_center
            length = vector.norm(dim=1)
            norm_vector = vector / length[:, None]
            mean_step = length / N_step
            train_rng = torch.arange(N_step)[None].float()

            instance_step = mean_step[:, None] * train_rng.repeat(pts_xyz.shape[0], 1)
            pts_xyz_all = pts_xyz[..., None, :] - norm_vector[..., None, :] * instance_step[..., None]

            total_empty_point = pts_xyz_all[:, 1:-1, :].flatten(0, 1)  # 搞一个mask 在这里边的就去掉



        if val_depth:
            # fix number
            max_sampled = max_depth * 1.40
            depth_stride = stride / 2

            N_step = int((max_sampled // depth_stride)) # 采样翻倍

            pts_xyz = torch.from_numpy(pcd_down_sample_point).float()
            vector = pts_xyz - all_cam_center
            length = vector.norm(dim=1)
            norm_vector = vector / length[:, None]
            mean_step = torch.tensor([depth_stride]).repeat(pts_xyz.shape[0])

            train_rng = torch.arange(N_step)[None].float()

            # pdb.set_trace()

            origin = torch.from_numpy(all_cam_center).repeat(pts_xyz.shape[0], 1)

            instance_step = mean_step[:, None] * train_rng.repeat(pts_xyz.shape[0], 1)
            pts_xyz_all = origin[..., None, :] + norm_vector[..., None, :] * instance_step[..., None]

            # get mask
            depth_mask = get_mask(pts_xyz_all, val_reso, max_height, max_depth)
            valid_pt = pts_xyz_all[~depth_mask]

            inputs['val_depth_empty'] = valid_pt
            inputs['mask'] = depth_mask
            inputs['total_depth_empty'] = pts_xyz_all
            inputs['surface_point'] = pts_xyz[..., :]
            inputs['origin'] = origin


        # check empty
        all_empty = total_empty_point
        # in_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(all_empty))  # 可以将车放置一些点作为voxel
        # out_mask = ~np.array(in_mask)
        # all_empty = all_empty[out_mask]


        # mask out colse to car
        # xy_range_1 = 2.0
        # xy_range_2 = 2.0
        # x
        mask7 = (all_empty[:, 0] < 3.5) & (all_empty[:, 0] > -0.5)  # 提前预设点
        # y
        mask8 = (all_empty[:, 1] < 1.0) & (all_empty[:, 1] > -1.0)
        mask = mask7 & mask8
        # o3d.visualization.draw_geometries([pcd])
        # # pdb.set_trace()

        all_empty = all_empty[~mask]
        # car_box = all_cam_center[np.newaxis, :]
        # pcd_car = o3d.geometry.PointCloud()
        # pcd_car.points = o3d.utility.Vector3dVector(car_box)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_car, voxel_size=1.5)
        # in_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(all_empty))  # 可以将车放置一些点作为voxel
        # out_mask = ~np.array(in_mask)
        # all_empty = all_empty[out_mask]

        # pdb.set_trace()
        inputs['all_empty'] = all_empty
        # print('empty size:', inputs['all_empty'].shape)
        # print('True_center size:', inputs['True_center'].shape)


        # val_label = {}
        # val_label['empty'] = inputs['empty']
        # val_label['True_center'] = inputs['True_center']
        # val_label['True_voxel_group'] = inputs['True_voxel_group']
        #
        # # val_label['True_vertex'] = inputs['True_vertex']
        # # val label
        # save_label_path = inputs['point_cloud_path'][0].replace('point_cloud_val', 'label/point_cloud_val_label_52_0.4')
        # (filepath, tempfilename) = os.path.split(save_label_path)
        # os.makedirs(filepath, exist_ok=True)
        # np.save(save_label_path, val_label)