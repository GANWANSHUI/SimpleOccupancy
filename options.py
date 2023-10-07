# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import configargparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:

    def __init__(self):
        self.parser = configargparse.ArgumentParser()

        self.parser.add_argument('--config', is_config_file=True,
                                 help='config file path')

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='./logs')


        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=34,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti"
                                 )
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=336)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=672)


        self.parser.add_argument("--height_ori",
                                 type=int,
                                 help="original input image height",
                                 default=1216)
        self.parser.add_argument("--width_ori",
                                 type=int,
                                 help="original input image width",
                                 default=1936)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=2.0)


        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load, currently only support for 3 frames",
                                 default=[0, -1, 1])


        self.parser.add_argument("--eval_only",
                                 help="if set, only evaluation",
                                 action="store_true")
        self.parser.add_argument("--use_fix_mask",
                                 help="if set, use self-occlusion mask (only for DDAD)",
                                 action="store_true")


        self.parser.add_argument("--spatial", type=lambda x: x.lower() == 'true', default=False,
                                 help="if set, use spatial photometric loss")


        self.parser.add_argument("--joint_pose",
                                 help="if set, use joint pose estimation",
                                 action="store_true")
        self.parser.add_argument("--model_type",
                                 type=str,
                                 default="unet")



        self.parser.add_argument("--use_sfm_spatial", type=lambda x: x.lower() == 'true', default=False,
                                 help="if set, use sfm pseudo label")


        self.parser.add_argument("--thr_dis",
                                 type=float,
                                 help="epipolar geometry threshold",
                                 default=1.0)
        self.parser.add_argument("--match_spatial_weight",
                                 type=float,
                                 help="sfm pretraining loss weight",
                                 default=0.1)
        self.parser.add_argument("--spatial_weight",
                                 type=float,
                                 help="spatial photometric loss weight",
                                 default=0.1)
        self.parser.add_argument("--skip",
                                 help="if set, use skip connection in CVT",
                                 action="store_true")

        self.parser.add_argument("--focal", type=lambda x: x.lower() == 'true', default=False,
                                 help="if set, use sfm pseudo label")

        self.parser.add_argument("--focal_scale",
                                 type=float,
                                 help="the global focal length to normalize depth",
                                 default=500)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=6)
        self.parser.add_argument("--B",
                                 type=int,
                                 help="real batch size",
                                 default=1)

        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)




        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=12)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet")


        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=20)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=25)

        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--eval_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1000)


        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        self.parser.add_argument("--local_rank", default=0, type=int)




        # customized
        self.parser.add_argument("--volume_depth",
                                 type=lambda x: x.lower() == 'true', default = True,
                                 help="if set, using the depth from volume rendering, rather than the depthdecoder", )

        self.parser.add_argument("--loss_type",  type=str,
                                 help="the loss for training [self, semi, gt]", default='gt')


        self.parser.add_argument("--trans2voxel", type=str,
                                 help="the manner from 2d feature to 3D voxel：[interpolation, transformer]",
                                 default = "interpolation")


        self.parser.add_argument("--voxels_size", type=int, action='append', default=[16, 256, 256],
                            help='the resolution of the voxel for rendering： Z, Y, X = 200, 8, 200')

        self.parser.add_argument("--real_size", type=int, action='append', default=[-52, 52, -52, 52, 0, 6],
                                 help='the real scale of the voxel: XMIN, XMAX, ZMIN, ZMAX, YMIN, YMAX')

        self.parser.add_argument("--scales",  action='append', type=int, help="scales used in the loss",
                                 default=[0])


        self.parser.add_argument("--stepsize",  help="stepsize for rendering",  type=float,  default=0.5)

        self.parser.add_argument("--en_lr", type=float, help="learning rate for encoder in volume rendering",
                                 default=0.0001)

        self.parser.add_argument("--de_lr", type=float, help="learning rate for decoder (3D CNN) in volume rendering", default=0.001)


        self.parser.add_argument("--aggregation", type=str,  help="the type of the feature aggregation [mlp 3dcnn 2dcnn]",default= '3dcnn')

        self.parser.add_argument("--pose_aug", type=str,
                                 help="do the augmentation on the camera pose for image level or car level [image, car, No]",
                                 default='No')

        self.parser.add_argument("--render_type", type=str,
                                 help="rednering by the density or probability [density, prob]", default='prob')

        self.parser.add_argument("--position", type=str,
                                 help="rednering by the density or probability [No, embedding, embedding1]",
                                 default='No')

        self.parser.add_argument("--data_type",  type=str,
                                 help=" data size for traing and testing - > [train_all, all, mini, tiny]",
                                 default='all')


        self.parser.add_argument("--log", type=lambda x: x.lower() == 'true', default = False,
                                 help="if set, using line space sample")

        self.parser.add_argument("--render_h", type=int, help="input image height",
                                 default=224)
        self.parser.add_argument("--render_w",  type=int,  help="input image width",
                                 default=352)

        self.parser.add_argument("--view_trans", type=str,
                                 help="the manner for image space to 3D volume space [simple, lift, bevformer]",
                                 default='simple')

        self.parser.add_argument("--input_channel", type=int, help="the final feature channel in the encoder",
                                 default=64)

        self.parser.add_argument("--con_channel", type=int, help="the final feature channel in the encoder",
                                 default=16)

        self.parser.add_argument("--out_channel", type=int, help="the output channel of the voxel",
                                 default=1)


        self.parser.add_argument("--cam_N", type=int, help="THE NUM OF CAM", default=6)


        self.parser.add_argument("--method", type=str,
                                 help="the method for the comparison [surrounddepth, CRF, monodepth2]",
                                 default='rendering')


        self.parser.add_argument("--encoder", type=str,
                                 help="the method for the comparison [101, 50]", default='50')


        self.parser.add_argument("--loss",  type=str,
                                 help="activation in the decoder [l1, sml1, silog, rl1]",
                                 default='silog')

        self.parser.add_argument("--evl_score", type=lambda x: x.lower() == 'true', default=True,
                                 help="if set, eval the occupancy score!")

        self.parser.add_argument("--surfaceloss", type=float, default=1.0,
                                 help="if tvloss > 0, using the  surface loss", )

        self.parser.add_argument("--empty_w", type=float, default=5.0,
                                 help="the weight of the empty point loss for the l1 grid loss", )

        self.parser.add_argument("--l1_voxel", type=str,
                                 help="activation in the decoder [No, ce, l1, ce_only, l1_only]", default='No')

        self.parser.add_argument("--val_reso", type=float, default=0.4, help="the resolution of the voxel in the evaluation [0.2, 0.4]")

        self.parser.add_argument("--N_trian", type=int, help="THE NUM OF sample point in the voxel training", default=30)
        self.parser.add_argument("--val_depth", type=lambda x: x.lower() == 'true', default=True, help="if set, do the depth voxel evaluation!")
        self.parser.add_argument("--use_t",  type=str, default='No', help="if [No, 2d, 3d], do the temporal information fusion!")

        self.parser.add_argument("--surround_view", type=lambda x: x.lower() == 'true', default=False, help="if set, eval the surrounding view depth!")

        self.parser.add_argument("--pretrain_path", type=str, help="pretrain path for the encoder", default='No')


        self.parser.add_argument("--ground_prior", type=lambda x: x.lower() == 'true', default=False,  help="if set, set the ground as 1!")

        self.parser.add_argument("--downsample_val", type=lambda x: x.lower() == 'true', default=True, help="if set")

        self.parser.add_argument("--val_binary", type=lambda x: x.lower() == 'true', default=False, help="if set")

        self.parser.add_argument("--abs", type=lambda x: x.lower() == 'true', default=False, help="if set, use the abs scale")

        self.parser.add_argument("--gt_pose", type=lambda x: x.lower() == 'true', default=False, help="if set, use the gt pose")

        self.parser.add_argument("--dataroot", type=str, help="the root for the ddad and nuscenes dataset", default='/data/ggeoinfo/Wanshui_BEV/data/ddad')



    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
