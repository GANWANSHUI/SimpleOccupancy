import pdb
import imageio
import open3d as o3d
import numpy as np
import os
import cv2

def render_mesh(show_list, W, H, intris, extr_list, should_collect_mesh_imgs = True, path=None):

    o3d_W, o3d_H = W, H

    for key, mesh2show in show_list.items():

        vis = o3d.visualization.Visualizer()

        vis.create_window(width=o3d_W, height=o3d_H, visible=False)
        vis.add_geometry(mesh2show)

        # pdb.set_trace()
        if key == 'gt':
            vis_ctrl = vis.get_render_option()
            vis_ctrl.mesh_show_back_face = True

        ctrl = vis.get_view_control()
        cam_model = ctrl.convert_to_pinhole_camera_parameters()

        # pdb.set_trace()
        # for surrounding view

        render_mesh_list = []

        for i in range (extr_list.shape[0]):
            # pdb.set_trace()
            intr = intris[i, ...]

            # cam_model.intrinsic.set_intrinsics(W, H, intr[0, 0], intr[1, 1], W / 2 - 0.5, H / 2 - 0.5)
            cam_model.intrinsic.set_intrinsics(W, H, intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2])
            extr = np.linalg.inv(extr_list[i, ...])
            cam_model.extrinsic = extr
            ctrl.convert_from_pinhole_camera_parameters(cam_model, True)

            vis.poll_events()
            vis.update_renderer()

            if should_collect_mesh_imgs:
                rgb_mesh = vis.capture_screen_float_buffer(do_render=False)
                rgb_mesh = (np.asarray(rgb_mesh) * 255.).clip(0, 255).astype(np.uint8)
                render_mesh_list.append(rgb_mesh)

        mesh_left_front_right = np.concatenate(
            (render_mesh_list[1], render_mesh_list[0], render_mesh_list[5]), axis=1)
        mesh_left_rear_right = np.concatenate(
            (render_mesh_list[2], render_mesh_list[3], render_mesh_list[4]), axis=1)

        mesh_surround_view = np.concatenate((mesh_left_front_right, mesh_left_rear_right), axis=0)

        imageio.imwrite(path.replace('mesh', 'mesh_vis/ground').replace('.ply', '_{}.png'.format(key)), mesh_surround_view)

        # vis.close()
        # for top view
        for i in range (1):
                # pdb.set_trace()
                intr = intris[i, ...]
                # cam_model.intrinsic.set_intrinsics(W, H, intr[0, 0], intr[1, 1], W / 2 - 0.5, H / 2 - 0.5)
                cam_model.intrinsic.set_intrinsics(W, H, intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2])
                extr = np.linalg.inv(extr_list[i, ...])
                cam_model.extrinsic = extr
                ctrl.convert_from_pinhole_camera_parameters(cam_model, True)

                # setting 1
                # 移动相机向上
                ctrl.translate(0.0, 200.0)
                # 相机朝下看
                ctrl.camera_local_rotate(0.0, 140.0)
                # 缩放场景
                ctrl.scale(8)

                # pdb.set_trace()
                # pip install open3d==0.16.0
                vis.poll_events()
                vis.update_renderer()

                if should_collect_mesh_imgs:
                    rgb_mesh = vis.capture_screen_float_buffer(do_render=False)
                    rgb_mesh = (np.asarray(rgb_mesh) * 255.).clip(0, 255).astype(np.uint8)
                    imageio.imwrite(path.replace('mesh', 'mesh_vis/top').replace('.ply', '_{}_top.png'.format(key)), rgb_mesh)

def save_video(image_path, FPS = 5 , W = 4800, H = 1856):

    all_img_list = []
    g = os.walk(image_path)
    for path, dir_list, file_list in g:
        # pdb.set_trace()
        # resort
        file_list.sort(key=lambda x: int(x.split('_')[0]))

        for file_name in file_list:
            all_img_list.append(os.path.join(path, file_name))

    CAMERA_FPS = FPS  # cam1.get(cv2.CAP_PROP_FPS)  # 25帧/秒
    # 定义视频编码格式
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    # 创建视频保存对象

    if 'ground' in image_path:
        videoWriter = cv2.VideoWriter(image_path.replace(r'mesh_vis\ground', r'mesh_vis') + r'\ground_video.mp4', fourcc, CAMERA_FPS, (W, H))
    else:
        videoWriter = cv2.VideoWriter(image_path.replace(r'mesh_vis\top', r'mesh_vis') + r'\top_video.mp4',
                                      fourcc, CAMERA_FPS, (W, H))
    # pdb.set_trace()

    print('begin writing!')
    for i in all_img_list:
        # pdb.set_trace()
        img = cv2.imread(i)
        #vpdb.set_trace()
        videoWriter.write(img)
        if 0xFF == ord('q'):
            break

    videoWriter.release()
    print('finish writing!')

    # pdb.set_trace()


def merge_image(root_path=None, FPS = 4, W = 4800, H = 1856):

    name = 'SDF'

    root_path = os.path.join('\\\?\\' + root_path)

    # mesh vis
    all_mesh_img_list = []
    mesh_img_path = os.path.join(root_path, r'mesh_vis\ground')
    g = os.walk(mesh_img_path)
    for path, dir_list, file_list in g:
        # resort
        file_list.sort(key=lambda x: int(x.split('_')[0]))
        for file_name in file_list:
            all_mesh_img_list.append(os.path.join(mesh_img_path, file_name))

    # rgb depth vis
    all_rgb_img_list = []
    rgb_img_path = os.path.join(root_path, 'visual_new')
    g = os.walk(rgb_img_path)
    for path, dir_list, file_list in g:
        # resort
        file_list.sort(key=lambda x: int(x.split('.')[0]))
        for file_name in file_list:
            all_rgb_img_list.append(os.path.join(rgb_img_path, file_name))

    # pdb.set_trace()
    CAMERA_FPS = FPS  # cam1.get(cv2.CAP_PROP_FPS)  # 25帧/秒
    # 定义视频编码格式
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    # 创建视频保存对象
    height = 300
    width = 672
    videoWriter = cv2.VideoWriter(root_path + r'\{}_rgb_depth_mesh.mp4'.format(name), fourcc,
                                  CAMERA_FPS, (3*width, 6*height))

    save_img_path = root_path + r'\rgb_depth_mesh_img'

    os.makedirs(save_img_path, exist_ok=True)


    for i in range (len(all_rgb_img_list)):
        rgb_img = cv2.imread(all_rgb_img_list[i])
        rgb_img = cv2.resize(rgb_img, (3*width, 4*height))

        mesh_img = cv2.imread(all_mesh_img_list[i])
        mesh_img = cv2.resize(mesh_img, (3 * width, 2 * height))
        # pdb.set_trace()
        cated_img = cv2.vconcat([rgb_img, mesh_img])

        cv2.imwrite(save_img_path+r'\{}.png'.format(i), cated_img)

        # pdb.set_trace()

        videoWriter.write(cated_img)
        if 0xFF == ord('q'):
            break

    videoWriter.release()
    print('finish writing!')


if __name__ == "__main__":

    merge_image()

# python D:\phd\project\bev\S3DO\utils\visual_tools.py