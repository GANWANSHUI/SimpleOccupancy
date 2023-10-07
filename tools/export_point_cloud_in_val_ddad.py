import os
import scipy.io as sio
import sys
from dgp.datasets import SynchronizedSceneDataset
import json
import random
from tqdm import tqdm
import copy
import numpy as np
import pdb
import pickle

root_path = '/data/ggeoinfo/Wanshui_BEV/data/ddad/raw_data'

dataset = SynchronizedSceneDataset(root_path + '/ddad.json',
                            datum_names=('CAMERA_01', 'lidar'),
                            generate_depth_from_datum='lidar',
                            split=sys.argv[1])

# pdb.set_trace()

root_path = '/data/ggeoinfo/Wanshui_BEV/data/ddad/point_cloud_train'
camera_names = ['lidar']

with open('../datasets/ddad/index.pkl', 'rb') as f:
    index_info = pickle.load(f)
to_save = {}

print(len(dataset))

for i in range(200):
    #for camera_name in camera_names:
    os.makedirs(os.path.join(root_path, '{:06d}'.format(i), 'LIDAR'), exist_ok=True)

count = 0


#
# with open('/home/wsgan/project/bev/SurroundDepth/datasets/ddad/info_val.pkl', 'rb') as f:
#     info = pickle.load(f)


for j in range (len(dataset)):

    data = dataset[j]
    count += 1
    print('count:', j)

    m = data[0][-1]  # 这里加载的data根据上边定义的传感器加载的

    n = data[0][0] # 获取camera timestamp
    t = str(n['timestamp']) # all the camera timestamp is the same, but the lidar is not the same.

    # pdb.set_trace()

    # print('pose', m['pose'])

    scene_id = index_info[t]['scene_name']

    save_temp = copy.deepcopy(m)

    if t not in to_save.keys():
        to_save[t] = copy.deepcopy(index_info[t])

    save_path = os.path.join(root_path, scene_id, m['datum_name'], t + '.npy')

    # save_dict = {}
    # pdb.set_trace()
    # save_dict['point_cloud'] = m['point_cloud']
    # save_dict['extrinsics'] = m['extrinsics'] 相機取的是extrinsics ， 不是pose
    # save_dict['pose'] = m['pose']

    # np.save(save_path, save_dict)
    np.save(save_path, m['point_cloud'])

print('finish')

# python /home/wsgan/project/bev/SurroundDepth/tools/export_point_cloud_in_val_ddad.py train

# python export_point_cloud_in_val_ddad.py train

# python export_point_cloud_in_val_ddad.py val

# 1.12.1+cu113 python 3.8.8