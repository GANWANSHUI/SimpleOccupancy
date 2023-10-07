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


with open(os.path.join(data_path, 'nuscenes_infos_val.pkl'), 'rb') as f:
    valdata = pickle.load(f)
    # len(valdata['infos']) 6019

with open(os.path.join(data_path, 'nuscenes_infos_train.pkl'), 'rb') as f:
    traindata = pickle.load(f)
    # len(traindata['infos']) 28130




for i in range(len(traindata)):

    pdb.set_trace()
    rec = nusc.get('sample', traindata[i]['token'])


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


# python /home/wsgan/project/bev/S3DO/tools/process_nusc_index.py

'''

with open('/mnt/cfs/algorithm/linqing.zhao/monodepth2/datasets/nusc/val_0.txt', 'w') as f:
    for info in val_list:
        f.writelines('{}\n'.format(info))   
'''