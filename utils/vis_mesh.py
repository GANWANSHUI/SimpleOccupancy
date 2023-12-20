from visual_tools import *

def vis_mesh(root_path):

    mesh_path_root = os.path.join('\\\?\\' + root_path + r'\mesh')
    # make list
    mesh_paths = os.listdir(mesh_path_root)

    mesh_paths.sort(key=lambda x: int(x.split('_')[0]))
    mesh_paths = [os.path.join(mesh_path_root, i) for i in mesh_paths]

    # mesh = o3d.io.read_triangle_mesh(path)
    # mesh.compute_vertex_normals() # compute normals for vertices or faces
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False)
    # pdb.set_trace()

    file_path = root_path + r'\visual_feature'
    file_path = os.path.join('\\\?\\' + file_path)
    file_list = os.listdir(file_path)
    file_list.sort(key=lambda x: int(x[:-4]))
    file_list = [os.path.join(file_path, i) for i in file_list]
    new_dict = np.load(file_list[0], allow_pickle='TRUE')
    data = dict(new_dict.item())

    extr_list = data['pose_spatial'].numpy()
    intris = data[('K', 0, 0)].numpy()

    W=672
    H=336

    # pdb.set_trace()
    for i in range (6):
        intris[i, 0, :] /= (W / 1600)
        intris[i, 1, :] /= (H / 928)

    W=1600
    H=928

    i = 0

    if 1:
        for path in mesh_paths:
            i += 1
            print(i)
            pred_geometry = o3d.io.read_triangle_mesh(path)
            pred_geometry.compute_vertex_normals()
            print('vis pred mesh')
            show_list = {'pred': pred_geometry}
            top_save_path = path.replace('mesh', r'mesh_vis\top')
            save_path_root = os.path.dirname(top_save_path)
            os.makedirs(save_path_root, exist_ok=True)

            # pdb.set_trace()
            ground_save_path = path.replace('mesh', r'mesh_vis\ground')
            save_path_root = os.path.dirname(ground_save_path)
            os.makedirs(save_path_root, exist_ok=True)
            render_mesh(show_list, W, H, intris, extr_list, path=path)


    video_path = mesh_path_root.replace('mesh', r'mesh_vis\ground')

    save_video(image_path = video_path)


if __name__ == "__main__":

    root_path = r'D:\phd\project\bev\SimpleOccupancy\logs\nuscenes\all_volume_True_loss_self_epoch_8_sdf_Yes\method_rendering_val_0.4_voxel_No_sur_1.0_empty_w_5.0_depth_52.0_out_1_en_50_input_64_vtrans_simple\step_0.5_size_256_rlde_0.001_aggregation_3dcnn_type_density_pe_No'

    vis_mesh(root_path)

    # refer this function to cat the RGB, Depth and mesh
    # merge_image()


# python D:\phd\project\bev\SimpleOccupancy\utils\vis_mesh.py