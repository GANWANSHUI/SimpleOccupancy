import os
import pdb

from dgp.datasets import SynchronizedSceneDataset
import copy
import numpy as np
import pickle
import sys

root_path = sys.argv[1]
save_point_cloud_path = os.path.split(root_path)[0] + '/point_cloud2023'

# train
dataset = SynchronizedSceneDataset(root_path + '/ddad.json',
                            datum_names=('CAMERA_01', 'lidar'),
                            generate_depth_from_datum='lidar',
                            split='train')

print('train length:{}'.format(len(dataset)))

with open('../datasets/ddad/index.pkl', 'rb') as f:
    index_info = pickle.load(f)

to_save = {}
for i in range(200):
    os.makedirs(os.path.join(save_point_cloud_path, '{:06d}'.format(i), 'LIDAR'), exist_ok=True)

count = 0

for j in range (len(dataset)):
    data = dataset[j]
    count += 1
    print('count:', j)

    m = data[0][-1]  # get lidar
    n = data[0][0] # get camera timestamp
    t = str(n['timestamp'])
    scene_id = index_info[t]['scene_name']

    save_temp = copy.deepcopy(m)

    if t not in to_save.keys():
        to_save[t] = copy.deepcopy(index_info[t])

    save_path = os.path.join(save_point_cloud_path, scene_id, m['datum_name'], t + '.npy')
    np.save(save_path, m['point_cloud'])


# val
dataset = SynchronizedSceneDataset(root_path + '/ddad.json',
                            datum_names=('CAMERA_01', 'lidar'),
                            generate_depth_from_datum='lidar',
                            split='val')
print('val length:{}'.format(len(dataset)))

for j in range (len(dataset)):
    data = dataset[j]
    count += 1
    print('count:', j)

    m = data[0][-1]  # get lidar
    n = data[0][0] # get camera timestamp
    t = str(n['timestamp'])
    scene_id = index_info[t]['scene_name']

    save_temp = copy.deepcopy(m)

    if t not in to_save.keys():
        to_save[t] = copy.deepcopy(index_info[t])

    save_path = os.path.join(save_point_cloud_path, scene_id, m['datum_name'], t + '.npy')
    np.save(save_path, m['point_cloud'])

print('finish ddad point cloud preparation!')


# cd /home/wsgan/project/bev/SimpleOccupancy/tools
# python export_point_cloud_ddad.py /data/ggeoinfo/Wanshui_BEV/data/ddad/raw_data
