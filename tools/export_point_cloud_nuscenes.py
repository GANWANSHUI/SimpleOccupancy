import os
import sys
# from py import process
import torch
import numpy as np

from PIL import Image
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import pdb
import imageio
import concurrent.futures as futures
from scipy.interpolate import griddata
import pickle



class DepthGenerator(object):
    def __init__(self):

        self.data_path = '/data/ggeoinfo/Wanshui_BEV/data/nuscenes/nuscenes'
        version = 'v1.0-trainval'

        self.nusc = NuScenes(version=version,
                            dataroot=self.data_path, verbose=False)

        with open('../datasets/nusc/{}.txt'.format(sys.argv[1]), 'r') as f:
            self.data = f.readlines()

        self.save_path = '/data/ggeoinfo/Wanshui_BEV/data/nuscenes/point_cloud_full'
        # self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

        # for camera_name in self.camera_names:
        #     os.makedirs(os.path.join(self.save_path, 'samples', camera_name), exist_ok=True)

    def __call__(self, num_workers=1): # 8
        print('generating nuscene depth maps from LiDAR projections')

        def process_one_sample(index):
            index_t = self.data[index].strip()
            rec = self.nusc.get('sample', index_t)

            lidar_sample = self.nusc.get(
                'sample_data', rec['data']['LIDAR_TOP'])
            lidar_pose = self.nusc.get(
                'ego_pose', lidar_sample['ego_pose_token'])
            #yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
            #lidar_rotation = Quaternion(scalar=np.cos(
            #    yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
            lidar_rotation= Quaternion(lidar_pose['rotation'])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])



            # get lidar points
            lidar_file = os.path.join(self.data_path, lidar_sample['filename'])
            lidar_points = np.fromfile(lidar_file, dtype=np.float32)
            # lidar data is stored as (x, y, z, intensity, ring index).
            lidar_points = lidar_points.reshape(-1, 5)[:, :4]

            # lidar points ==> ego frame
            sensor_sample = self.nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            lidar_to_ego_lidar_rot = Quaternion(sensor_sample['rotation']).rotation_matrix
            lidar_to_ego_lidar_trans = np.array(sensor_sample['translation']).reshape(1, 3)

            ego_lidar_points = np.dot(lidar_points[:, :3], lidar_to_ego_lidar_rot.T)
            ego_lidar_points += lidar_to_ego_lidar_trans

            homo_ego_lidar_points = np.concatenate((ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)

            homo_ego_lidar_points = torch.from_numpy(homo_ego_lidar_points).float()


            for cam in self.camera_names:
                camera_sample = self.nusc.get('sample_data', rec['data'][cam])
                car_egopose = self.nusc.get(
                    'ego_pose', camera_sample['ego_pose_token'])
                egopose_rotation = Quaternion(car_egopose['rotation']).inverse
                egopose_translation = - \
                    np.array(car_egopose['translation'])[:, None]
                world_to_car_egopose = np.vstack([
                    np.hstack((egopose_rotation.rotation_matrix,
                               egopose_rotation.rotation_matrix @ egopose_translation)),
                    np.array([0, 0, 0, 1])
                ])

                print('egopose_rotation', egopose_rotation)
                print('egopose_translation', egopose_translation)
                print('world_to_car_egopose', world_to_car_egopose)



            pdb.set_trace()
            # np.save(os.path.join(self.save_path, point_cloud_name + '.npy'), homo_ego_lidar_points)

            # 将点云存成当前帧 CAM_FRONT 同名
            print('finish processing index = {:06d}'.format(index))


        sample_id_list = list(range(len(self.data)))
        with futures.ThreadPoolExecutor(num_workers) as executor:
            executor.map(process_one_sample, sample_id_list)


if __name__ == "__main__":
    model = DepthGenerator()
    model()


# python export_point_cloud_nuscenes.py train
# python export_point_cloud_nuscenes.py val