import open3d as o3d
import numpy as np
import pdb, os


file_path = r'D:\phd\project\bev\SurroundDepth\data\ddad\point_cloud_val\000150\LIDAR\15616458249936530.npy'
save_path = file_path.replace('point_cloud_val', 'point_cloud_voxel_val')

point_cloud = np.load(file_path)



# voxel_size = [0.5, 0.75, 1.0]
voxel_size = [0.5, 1.0]


def get_occupancy_label(all_cam_center, pts_xyz, voxel_size):

    occupancy_label = dict()

    # 过滤点云

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz)
    # o3d.visualization.draw_geometries([pcd])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    # data['voxel'] = dict(xyzrgb=dict(type=Vis3DType.voxel, data=voxel_grid, visible=True, point_size=1))

    total_vertex = []
    total_center = []
    total_voxel_group = [] # we define a voxel's 8 vertex and its center as a group

    # get vertex and center point
    for i in range(pts_xyz.shape[0]):
        print('get vertex and center point: ', i)

        instance_group = []

        # get vertex
        vertex = np.asarray(voxel_grid.get_voxel_bounding_points(voxel_grid.get_voxel(pts_xyz[i, :])))
        for j in range(8):
            vertexlist = vertex[j].tolist()
            instance_group.append(vertexlist)

            # if vertexlist not in total_vertex:
            total_vertex.append(vertexlist)

        # pdb.set_trace()
        # get voxel center
        center_point = voxel_grid.get_voxel_center_coordinate(voxel_grid.get_voxel(pts_xyz[i, :]))
        centerlist = center_point.tolist()

        # if centerlist not in total_center:
        total_center.append(centerlist)

        instance_group.append(centerlist)

        # if instance_group not in total_voxel_group: # 需要注意有无量化的误差，导致误判
        total_voxel_group.append(instance_group)

    # 全部添加，然后remove 重复的 转numpy
    # delete repeat items
    # total_voxel_group = list(set(total_voxel_group))

    # 转numpy, 然后去除重复的voxel
    total_voxel_group = np.array(total_voxel_group)
    total_voxel_group = np.unique(total_voxel_group, axis = 0)

    total_center = np.array(total_center)
    total_center = np.unique(total_center, axis=0)

    total_vertex = np.array(total_vertex)
    total_vertex = np.unique(total_vertex, axis=0)


    occupancy_label['True_center'] = np.array(total_center)
    occupancy_label['True_vertex'] = np.array(total_vertex)
    occupancy_label['True_voxel_group'] = np.array(total_voxel_group)

    print('total vertex {}'.format(len(voxel_grid.get_voxels()) * 8))
    print('total filtered vertex {}'.format(len(total_vertex)))
    print('total center {}'.format(len(total_center)))
    print('total voxel group {}'.format(len(total_voxel_group)))


    # get empty point
    total_empty_point = []

    for i in range(len(total_center)):
        stride = voxel_size
        vector = np.array(total_center[i]) - np.array(all_cam_center)
        length = np.linalg.norm(vector)
        norm_vector = vector / length

        print('calculating the empty point:', i)
        # pdb.set_trace()
        for j in range(1, int((length // stride) - (1.5 // stride))):
            sampled_point = total_center[i] - stride * j * norm_vector
            sampled_point = sampled_point.tolist()
            # if sampled_point not in total_empty_point:
            total_empty_point.append(sampled_point)

    # check empty
    all_empty = np.array(total_empty_point)
    in_mask = voxel_grid.check_if_included(o3d.utility.Vector3dVector(all_empty))
    out_mask = ~np.array(in_mask)
    revised_empty = all_empty[out_mask.tolist()]

    check_empty = np.array(revised_empty)
    if np.array(voxel_grid.check_if_included(o3d.utility.Vector3dVector(check_empty))).max():
        print('error')

    pts_empty = check_empty
    occupancy_label['empty'] = np.array(pts_empty)

    return occupancy_label


pose = np.array([[[ 5.2637e-02, -7.3505e-03,  9.9859e-01,  1.6784e+00],
         [-9.9859e-01, -6.5109e-03,  5.2590e-02,  2.7305e-01],
         [ 6.1151e-03, -9.9995e-01, -7.6829e-03,  1.5483e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
        [[ 8.1031e-01,  1.7995e-03,  5.8601e-01,  1.6798e+00],
         [-5.8585e-01, -2.0823e-02,  8.1015e-01,  4.3545e-01],
         [ 1.3660e-02, -9.9978e-01, -1.5818e-02,  1.5112e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 8.6259e-01,  5.5015e-03, -5.0588e-01,  1.3349e+00],
         [ 5.0582e-01, -2.7598e-02,  8.6220e-01,  4.8815e-01],
         [-9.2178e-03, -9.9960e-01, -2.6589e-02,  1.5116e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-1.5745e-02,  2.0812e-02, -9.9966e-01,  3.7783e-01],
         [ 9.9903e-01, -4.0711e-02, -1.6582e-02,  1.6531e-01],
         [-4.1043e-02, -9.9895e-01, -2.0151e-02,  1.4763e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-8.5554e-01,  1.1021e-02, -5.1763e-01,  1.3293e+00],
         [ 5.1761e-01, -4.9973e-03, -8.5560e-01, -4.1497e-01],
         [-1.2016e-02, -9.9993e-01, -1.4291e-03,  1.5565e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-7.8927e-01, -3.2821e-03,  6.1404e-01,  1.7405e+00],
         [-6.1394e-01, -1.4644e-02, -7.8922e-01, -4.4302e-01],
         [ 1.1582e-02, -9.9989e-01,  9.5432e-03,  1.5890e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]])

all_pose = pose
x = 0
y = 0
z = 0
for k in range(6):
    x += all_pose[k][0][3]
    y += all_pose[k][1][3]
    z += all_pose[k][2][3]

all_cam_center = [x / 6, y / 6, z / 6]


instance_voxel = dict()

GT_point = point_cloud[:, :] # down sample


# GT预处理 8 > z > 0, -200 < (x,y) < 200
# z
mask1 = GT_point[:, 2] < 0.2  # 因为voxel，地面为1/2分辨率： 0.5 -> 0.25
mask2 = GT_point[:, 2] > 7

xy_range = 80

# x
mask3 = GT_point[:, 0] > xy_range
mask4 = GT_point[:, 0] < -xy_range

# y
mask5 = GT_point[:, 1] > xy_range
mask6 = GT_point[:, 1] < -xy_range

mask = mask1 + mask2 + mask3 + mask4 + mask5 + mask6
GT_point = GT_point[~mask]


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(GT_point)
downpcd = pcd.voxel_down_sample(voxel_size=0.4)

# pdb.set_trace()
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([downpcd])

GT_point = np.asarray(downpcd.points)


for size in voxel_size:

    label = get_occupancy_label(all_cam_center, GT_point, size)
    instance_voxel['{}'.format(size)] = label

# pdb.set_trace()
# os.makedirs(save_path, exist_ok=True)
np.save(save_path, instance_voxel)

voxel_path = r'D:\phd\project\bev\SurroundDepth\data\ddad\point_cloud_voxel_val\000150\LIDAR\15616458249936530.npy'
voxel_1 = np.load(voxel_path, allow_pickle=True)
voxel_1 = dict(voxel_1.item())
pdb.set_trace()




