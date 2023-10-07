import os
import scipy.io as sio
import sys
# sys.path.append('/home/linqing.zhao/dgp')
#from dgp.datasets import SynchronizedSceneDataset
import json
import random
import pickle
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np
import pdb


data_path = '/data/ggeoinfo/Wanshui_BEV/data/nuscenes/nuscenes/'
version = 'v1.0-trainval'
nusc = NuScenes(version=version,
                dataroot=data_path, verbose=False)
train_list, val_list = [], []

# with open(os.path.join(data_path, 'nuscenes_infos_10sweeps_val.pkl'), 'rb') as f:
#     valdata = pickle.load(f)
#     # len(valdata) 6019
#
# with open(os.path.join(data_path, 'nuscenes_infos_10sweeps_train.pkl'), 'rb') as f:
#     traindata = pickle.load(f)
#     # len(traindata) 28130


# with open(os.path.join(data_path, 'nuscenes_infos_val.pkl'), 'rb') as f:
#     valdata = pickle.load(f)
#
#     # len(valdata['infos']) 6019
#     with open(os.path.join(data_path, 'nuscenes_occupancy_flow_infos_val.pkl'), "wb") as tf:
#         save_dict = {}
#         for i in range(len(valdata['infos'])):
#             print('i:{}'.format(i))
#             save_dict[valdata['infos'][i]['token']] = valdata['infos'][i]
#
#         pickle.dump(save_dict, tf)


with open(os.path.join(data_path, 'nuscenes_infos_train.pkl'), 'rb') as f:
    traindata = pickle.load(f)
    # len(traindata['infos']) 28130

    for j in range (len(traindata['infos'])):
        save_dict = {}

        instance_scene = traindata['infos'][j]

        # info
        save_dict[instance_scene['token']] = instance_scene # 这个作为保存的名字: labeled point cloud (x, y, z, category, velocity) + info -> npy


        lidar_path = instance_scene['lidar_path'].replace('raw_data', 'nuscenes')
        all_pts = np.fromfile(lidar_path, dtype=np.float32)

        for i in range (instance_scene['gt_names'].shape[0]):

            # get class label
            gt_names = instance_scene['gt_names'][i] # 先根据目标检测的顺序定义一个list, 然后去index

            # get point label
            gt_boxes = instance_scene['gt_boxes'][i] # 3D bounding box (gravity) center location (3), size (3), (global) yaw angle (1)
            #
            min_point = 2
            max_point = 1

            # get the point inside the bbx
            mask_x = (all_pts[:, 0] < 3.5) & (all_pts[:, 0] > -0.5) # x
            mask_y = (all_pts[:, 1] < 1.0) & (all_pts[:, 1] > -1.0) # y
            mask_z = (all_pts[:, 2] < 1.0) & (all_pts[:, 2] > -1.0) # z

            mask = mask_x & mask_y & mask_z


            gt_velocity = instance_scene['gt_velocity'][i]
            num_lidar_pts = instance_scene['num_lidar_pts'][i] # 这个来验证框选取的点对不对


            pdb.set_trace()


    # with open(os.path.join(data_path, 'nuscenes_occupancy_flow_infos_train.pkl'), "wb") as tf:
    #     save_dict = {}
    #     # pdb.set_trace()
    #     for i in range(len(traindata['infos'])):
    #         save_dict[traindata['infos'][i]['token']] = traindata['infos'][i]
    #         print('i:{}'.format(i))
    #
    #     pickle.dump(save_dict, tf)

exit()

for i in range(len(traindata)):

    pdb.set_trace()

    # traindata['infos'][0] ->
    # dict_keys(['lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag'])

    save_dict[traindata['infos'][i]['token']] = traindata['infos'][i]
    continue


    # traindata['metadata']['version']
    rec = nusc.get('sample', traindata[i]['token'])  # scene level

    sample_annotation = nusc.get('sample_annotation', rec['anns'][0])




    # 过滤没有前一帧或者后一帧的数据
    if rec['prev'] == '' or rec['next'] == '':
        continue





    cam_sample = nusc.get('sample_data', rec['data']['CAM_FRONT_LEFT'])
    ego = nusc.get('ego_pose', cam_sample['ego_pose_token'])
    pose = Quaternion(ego['rotation']).transformation_matrix
    pose[:3, 3] = np.array(ego['translation'])

    rec_next = nusc.get('sample', rec['next'])
    cam_sample_next = nusc.get('sample_data', cam_sample['next']) #rec_next['data']['CAM_FRONT_LEFT'])
    ego_next = nusc.get('ego_pose', cam_sample_next['ego_pose_token'])
    pose_next = Quaternion(ego_next['rotation']).transformation_matrix
    pose_next[:3, 3] = np.array(ego_next['translation'])

    cam_sample_prev = nusc.get('sample_data', cam_sample['prev']) #rec_next['data']['CAM_FRONT_LEFT'])
    ego_prev = nusc.get('ego_pose', cam_sample_prev['ego_pose_token'])
    pose_prev = Quaternion(ego_prev['rotation']).transformation_matrix
    pose_prev[:3, 3] = np.array(ego_prev['translation'])

    # print(pose_prev[2][-1])

    # pdb.set_trace()
    # # LIDAR_TOP
    cam_sample = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    ego = nusc.get('ego_pose', cam_sample['ego_pose_token'])
    pose = Quaternion(ego['rotation']).transformation_matrix
    pose[:3, 3] = np.array(ego['translation'])



    #print(np.linalg.norm(pose_next[:3, 3] - pose[:3, 3]))
    # 过滤 if the scale of translation between current frame and previous frame or current frame and next frame is smaller than 0.1m
    # if np.linalg.norm(pose_next[:3, 3] - pose[:3, 3]) < 0.1 or np.linalg.norm(pose_prev[:3, 3] - pose[:3, 3]) < 0.1:
    #     continue

    train_list.append(traindata[i]['token'])



#
j = 0
for i in range(len(valdata)):

    rec = nusc.get(
            'sample', valdata[i]['token'])

    # 过滤没有前一帧或者后一帧的数据
    if rec['prev'] == '' or rec['next'] == '':
        j +=1
        print(j)
        # print('None')
        continue

    val_list.append(valdata[i]['token'])

# with open('/home/wsgan/project/bev/SurroundDepth/datasets/nusc/val_with_prev_next.txt', 'w') as f:
#     random.shuffle(val_list)
#     for info in val_list:
#         #f.writelines('{} {}\n'.format(info[0], info[1]))
#         f.writelines('{}\n'.format(info))




# python /home/wsgan/project/bev/S3DO/tools/nuscene_label_transformer.py

'''

with open('/mnt/cfs/algorithm/linqing.zhao/monodepth2/datasets/nusc/val_0.txt', 'w') as f:
    for info in val_list:
        f.writelines('{}\n'.format(info))   
'''