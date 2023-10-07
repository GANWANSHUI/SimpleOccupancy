# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import pdb
import time
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil
import json


import datasets
import networks

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
from torch.utils.data import DistributedSampler as _DistributedSampler
import pickle

import sys
sys.path.append("..")
from utils.loss_metric import *
from utils.layers import *
import utils.basic as basic


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)


class Runer:

    def __init__(self, options):

        self.opt = options
        self.opt.B = self.opt.batch_size // 6

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name +
                                     '{}_volume_{}_loss_{}_epoch_{}/method_{}_val_{}_voxel_{}_sur_{}_empty_w_{}_depth_{}_out_{}_en_{}_input_{}_vtrans_{}/step_{}_size_{}_rlde_{}_aggregation_{}_type_{}_pe_{}'.format(
                                         self.opt.data_type, self.opt.volume_depth, self.opt.loss_type, self.opt.num_epochs,
                                         self.opt.method, self.opt.val_reso, self.opt.l1_voxel,
                                         self.opt.surfaceloss, self.opt.empty_w, self.opt.max_depth,
                                         self.opt.out_channel, self.opt.encoder, self.opt.input_channel, self.opt.view_trans,
                                         self.opt.stepsize, self.opt.voxels_size[2], self.opt.de_lr,
                                         self.opt.aggregation, self.opt.render_type, self.opt.position))


        print('log path:', self.log_path)
        os.makedirs(os.path.join(self.log_path, 'eval'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'visual_rgb_depth'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'visual_feature'), exist_ok=True)

        # pdb.set_trace()

        self.models = {}
        self.parameters_to_train = []

        self.local_rank = self.opt.local_rank
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend='nccl')
        self.device = torch.device("cuda", self.local_rank)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])


        self.opt.focal = False
        self.models["encoder"] = networks.Encoder_res101(self.opt.input_channel, path=None, network_type=self.opt.encoder)
        self.models["depth"] = networks.VolumeDecoder(self.opt)


        self.models["encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["encoder"])
        self.models["encoder"] = (self.models["encoder"]).to(self.device)

        self.parameters_to_train += [{'params': self.models["encoder"].parameters(), 'lr': self.opt.learning_rate}]

        self.models["depth"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["depth"])
        self.models["depth"] = (self.models["depth"]).to(self.device)


        # pdb.set_trace()
        # self.parameters_to_train += [{'params': self.models["depth"].parameters(), 'lr': self.opt.de_lr}]

        if self.opt.position == 'embedding1':
            self.parameters_to_train += [{'params': self.models["depth"]._3DCNN.parameters(), 'lr': self.opt.de_lr}, {'params': self.models["depth"].embeddingnet.parameters(), 'lr': self.opt.en_lr}]
        else:
            self.parameters_to_train += [{'params': self.models["depth"].parameters(), 'lr': self.opt.de_lr}]

        self.silog_criterion = silog_loss(variance_focus=0.85)


        if self.opt.load_weights_folder is not None:
            self.load_model()

        for key in self.models.keys():
            self.models[key] = DDP(self.models[key], device_ids=[self.local_rank], output_device=self.local_rank,
                                   find_unused_parameters=True, broadcast_buffers=False)

        self.model_optimizer = optim.Adam(self.parameters_to_train)
        self.criterion = nn.BCELoss()
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, gamma = 0.1, last_epoch=-1)


        for key in self.models.keys():
            for name, param in self.models[key].named_parameters():
                if param.requires_grad:
                    pass
                else:
                    print(name)
                    # print(param.data)
                    print("requires_grad:", param.requires_grad)
                    print("-----------------------------------")

        if self.local_rank == 0:
            self.log_print("Training model named: {}".format(self.opt.model_name))

        # pdb.set_trace()
        # data
        datasets_dict = {"ddad": datasets.DDADDataset,
                         "nusc": datasets.NuscDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        self.opt.batch_size = self.opt.batch_size // 6


        train_dataset = self.dataset(self.opt,
                                     self.opt.height, self.opt.width,
                                     self.opt.frame_ids, num_scales=self.num_scales, is_train=True,
                                     volume_depth=self.opt.volume_depth)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

        # pdb.set_trace()
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, collate_fn=self.my_collate,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

        self.num_total_steps = len(self.train_loader) * self.opt.num_epochs

        val_dataset = self.dataset(self.opt,
                                   self.opt.height, self.opt.width,
                                   self.opt.frame_ids, num_scales=1, is_train=False, volume_depth=self.opt.volume_depth)


        rank, world_size = get_dist_info()
        self.world_size = world_size
        val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=False)


        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, collate_fn=self.my_collate,
            num_workers=15, pin_memory=True, drop_last=False, sampler=val_sampler)


        self.val_iter = iter(self.val_loader)
        self.num_val = len(val_dataset)

        self.opt.batch_size = self.opt.batch_size * 6
        self.num_val = self.num_val * 6

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)


        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if self.local_rank == 0:
            self.log_print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))

        self.save_opts()


    def my_collate(self, batch):
        batch_new = {}
        keys_list = list(batch[0].keys())
        special_key_list = ['id', 'match_spatial', 'point_cloud', 'label', 'point_cloud_path', 'point_cloud_label']

        for key in keys_list:
            if key not in special_key_list:
                # print('key:', key)
                batch_new[key] = [item[key] for item in batch]
                try:
                    batch_new[key] = torch.cat(batch_new[key], axis=0)
                except:
                    print('key', key)

            else:
                batch_new[key] = []
                for item in batch:
                    for value in item[key]:
                        # print(value.shape)
                        batch_new[key].append(value)

        return batch_new

    def to_device(self, inputs):

        special_key_list = ['id', 'label', 'point_cloud', 'label_empty', 'label_center', 'empty', 'True_center',
                            'all_empty',
                            'True_voxel_group', 'point_cloud_path', 'point_cloud_label', ('K_ori', -1), ('K_ori', 1)]
        match_key_list = ['match_spatial']

        for key, ipt in inputs.items():

            if key in special_key_list:
                inputs[key] = ipt

            elif key in match_key_list:
                for i in range(len(inputs[key])):
                    inputs[key][i] = inputs[key][i].to(self.device)
            else:
                inputs[key] = ipt.to(self.device)

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline"""
        if self.local_rank == 0:

            os.makedirs(os.path.join(self.log_path, 'code'), exist_ok=True)

            # back up files
            source1 = './runer.py'
            source3 = './run.py'
            source4 = './options.py'
            source13 = './run.sh'

            source6 = './configs'
            source7 = './networks'
            source8 = './datasets'
            source9 = './utils'

            source = [source1, source3, source4, source13]
            for i in source:
                shutil.copy(i, os.path.join(self.log_path, 'code'))

            if not os.path.exists(os.path.join(self.log_path, 'code' + '/configs')):
                shutil.copytree(source6, os.path.join(self.log_path, 'code' + '/configs'))

            if not os.path.exists(os.path.join(self.log_path, 'code' + '/networks')):
                shutil.copytree(source7, os.path.join(self.log_path, 'code' + '/networks'))

            if not os.path.exists(os.path.join(self.log_path, 'code' + '/datasets')):
                shutil.copytree(source8, os.path.join(self.log_path, 'code' + '/datasets'))

            if not os.path.exists(os.path.join(self.log_path, 'code' + '/utils')):
                shutil.copytree(source9, os.path.join(self.log_path, 'code' + '/utils'))

        self.step = 1

        if self.opt.eval_only:
            self.val()
            if self.local_rank == 0:
                self.evaluation(evl_score=True)

            return None

        self.epoch = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.train_loader.sampler.set_epoch(self.epoch)
            self.run_epoch()

        self.save_model()

        self.val()

        if self.local_rank == 0:
            self.evaluation(evl_score=True)

        return None

    def evaluation(self, evl_score=False):

        batch_size = self.world_size

        if self.local_rank == 0:
            self.log_print("-> Evaluating {} in {}".format('final', batch_size))
            # for occupancy metric
            if self.opt.evl_score and evl_score:
                no_group_score = []
                group_score = []
                time.sleep(10)
                for i in range(batch_size):
                    while not os.path.join(self.log_path, 'eval', '{}_occupancy.npy'.format(i)):
                        time.sleep(15)
                    new_dict = np.load(os.path.join(self.log_path, 'eval', '{}_occupancy.npy'.format(i)),
                                       allow_pickle='TRUE')
                    data = dict(new_dict.item())

                    no_group_score.append(data['no_group'])

                merge_no_group_score = no_group_score[0]
                for idx, val in enumerate(no_group_score):
                    if idx == 0:
                        pass
                    else:
                        merge_no_group_score = np.concatenate([merge_no_group_score, no_group_score[idx]], axis=1)
                # pdb.set_trace()
                self.log_print('\n effective occupancy cases: {} \n'.format(merge_no_group_score.shape[1]))
                merge_no_group_score = np.array(merge_no_group_score).mean(1)


                for idx, val in enumerate(data['metric']):
                    self.log_print('no group {} :\n{}'.format(val, np.around(merge_no_group_score[idx], 4)))
                    # self.log_print('group {} :\n{}\n'.format(val, np.around(group_mean[idx], 4)))
                    self.log_print('-----------------------------------------------------')

            errors = {}
            eval_types = ['scale-ambiguous', 'scale-aware']
            for eval_type in eval_types:
                errors[eval_type] = {}

            for i in range(batch_size):
                while not os.path.exists(os.path.join(self.log_path, 'eval', '{}.pkl'.format(i))):
                    time.sleep(10)
                time.sleep(5)
                with open(os.path.join(self.log_path, 'eval', '{}.pkl'.format(i)), 'rb') as f:
                    errors_i = pickle.load(f)
                    for eval_type in eval_types:
                        for camera_id in errors_i[eval_type].keys():
                            if camera_id not in errors[eval_type].keys():
                                errors[eval_type][camera_id] = []

                            errors[eval_type][camera_id].append(errors_i[eval_type][camera_id])

            num_sum = 0
            for eval_type in eval_types:
                for camera_id in errors[eval_type].keys():
                    errors[eval_type][camera_id] = np.concatenate(errors[eval_type][camera_id], axis=0)

                    if eval_type == 'scale-aware':
                        num_sum += errors[eval_type][camera_id].shape[0]

                    errors[eval_type][camera_id] = errors[eval_type][camera_id].mean(0)

            for eval_type in eval_types:
                self.log_print("{} evaluation:".format(eval_type))
                mean_errors_sum = 0
                for camera_id in errors[eval_type].keys():
                    mean_errors_sum += errors[eval_type][camera_id]
                mean_errors_sum /= len(errors[eval_type].keys())
                errors[eval_type]['all'] = mean_errors_sum

                for camera_id in errors[eval_type].keys():
                    mean_errors = errors[eval_type][camera_id]
                    self.log_print(camera_id)
                    self.log_print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                    self.log_print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

            assert num_sum == self.num_val


    def val(self, save_image=True):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        errors = {}
        eval_types = ['scale-ambiguous', 'scale-aware']
        for eval_type in eval_types:
            errors[eval_type] = {}

        self.models["encoder"].eval()
        self.models["depth"].eval()
        ratios_median = []

        print('begin eval!')
        total_time = []

        total_abs_rel_26 = []
        total_sq_rel_26 = []
        total_rmse_26 = []
        total_rmse_log_26 = []
        total_a1_26 = []
        total_a2_26 = []
        total_a3_26 = []

        # depth occuapncy
        total_abs_rel_52 = []
        total_sq_rel_52 = []
        total_rmse_52 = []
        total_rmse_log_52 = []
        total_a1_52 = []
        total_a2_52 = []
        total_a3_52 = []


        total_evl_time = time.time()

        with torch.no_grad():
            loader = self.val_loader
            for idx, data in enumerate(loader):

                eps_time = time.time()

                input_color = data[("color", 0, 0)].cuda()

                gt_depths = data["depth"].cpu().numpy()
                camera_ids = data["id"]

                features = self.models["encoder"](input_color)

                output = self.models["depth"](features, data, is_train=False)

                eps_time = time.time() - eps_time
                total_time.append(eps_time)

                if self.opt.volume_depth and self.opt.evl_score:

                    threshold_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
                    xy_range = [26, 52]

                    for x_range_i in xy_range:
                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, point_error = get_occupancy_depth_score(inputs=data, outputs=output, threshold_list=threshold_list, xy_range = x_range_i)

                        if x_range_i == 52:
                            total_abs_rel_52.append(abs_rel)
                            total_sq_rel_52.append(sq_rel)
                            total_rmse_52.append(rmse)
                            total_rmse_log_52.append(rmse_log)
                            total_a1_52.append(a1)
                            total_a2_52.append(a2)
                            total_a3_52.append(a3)

                        else:
                            total_abs_rel_26.append(abs_rel)
                            total_sq_rel_26.append(sq_rel)
                            total_rmse_26.append(rmse)
                            total_rmse_log_26.append(rmse_log)
                            total_a1_26.append(a1)
                            total_a2_26.append(a2)
                            total_a3_26.append(a3)


                    if self.local_rank == 0:
                        print('abs_rel:', np.around(abs_rel, 4))

                if self.local_rank == 0:
                    print('single inference:(eps time:', eps_time, 'secs)')

                if self.opt.volume_depth:
                    pred_disps_flip = output[("disp", 0)]


                pred_disps = pred_disps_flip.cpu()[:, 0].numpy()

                concated_image_list = []
                concated_depth_list = []

                for i in range(pred_disps.shape[0]):

                    camera_id = camera_ids[i]

                    if camera_id not in list(errors['scale-aware']):
                        errors['scale-aware'][camera_id] = []
                        errors['scale-ambiguous'][camera_id] = []

                    gt_depth = gt_depths[i]
                    gt_height, gt_width = gt_depth.shape[:2]

                    pred_disp = pred_disps[i]

                    if self.opt.volume_depth:
                        pred_depth = pred_disp

                        if self.local_rank == 0:
                            print('volume rendering depth: min {}, max {}'.format(pred_depth.min(), pred_depth.max()))

                    pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))


                    mask = np.logical_and(gt_depth > self.opt.min_depth, gt_depth < self.opt.max_depth)

                    if self.local_rank == 0:
                        pred_depth_color = visualize_depth(pred_depth.copy())
                        color = (input_color[i].cpu().permute(1, 2, 0).numpy()) * 255
                        color = color[..., [2, 1, 0]]

                        concated_image_list.append(color)
                        concated_depth_list.append(cv2.resize(pred_depth_color.copy(), (self.opt.width, self.opt.height)))

                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]

                    ratio_median = np.median(gt_depth) / np.median(pred_depth)
                    ratios_median.append(ratio_median)
                    pred_depth_median = pred_depth.copy() * ratio_median

                    pred_depth_median[pred_depth_median < self.opt.min_depth] = self.opt.min_depth
                    pred_depth_median[pred_depth_median > self.opt.max_depth] = self.opt.max_depth

                    errors['scale-ambiguous'][camera_id].append(compute_errors(gt_depth, pred_depth_median))

                    pred_depth[pred_depth < self.opt.min_depth] = self.opt.min_depth
                    pred_depth[pred_depth > self.opt.max_depth] = self.opt.max_depth

                    errors['scale-aware'][camera_id].append(compute_errors(gt_depth, pred_depth))


                save_num = 10
                if self.opt.eval_only:
                    save_num = 10

                if self.local_rank == 0 and save_image and idx < save_num:
                    print('idx:', idx)

                    image_left_front_right = np.concatenate(
                        (concated_image_list[1], concated_image_list[0], concated_image_list[5]), axis=1)
                    image_left_rear_right = np.concatenate(
                        (concated_image_list[2], concated_image_list[3], concated_image_list[4]), axis=1)

                    image_surround_view = np.concatenate((image_left_front_right, image_left_rear_right), axis=0)

                    depth_left_front_right = np.concatenate(
                        (concated_depth_list[1], concated_depth_list[0], concated_depth_list[5]), axis=1)
                    depth_left_rear_right = np.concatenate(
                        (concated_depth_list[2], concated_depth_list[3], concated_depth_list[4]), axis=1)

                    depth_surround_view = np.concatenate((depth_left_front_right, depth_left_rear_right), axis=0)
                    surround_view = np.concatenate((image_surround_view, depth_surround_view), axis=0)

                    # pdb.set_trace()
                    cv2.imwrite('{}/visual_rgb_depth/{}.jpg'.format(self.log_path, idx), surround_view)


                    if self.opt.evl_score and save_image and self.opt.eval_only:
                        vis_dic = {}
                        vis_dic['depth_color'] = concated_depth_list
                        vis_dic['rgb'] = concated_image_list
                        vis_dic['point_cloud'] = data['point_cloud']

                        # for direct vis
                        vis_dic['pred_True_center'] = output[('True_center')]
                        vis_dic['pred_empty'] = output[('empty')]

                        vis_dic['pts_True_center'] = data[('True_center')]
                        vis_dic['pts_empty'] = data[('all_empty')]
                        vis_dic['pose_spatial'] = data['pose_spatial']
                        vis_dic['opt'] = self.opt

                        vis_dic['probability'] = output['density']

                        np.save('{}/visual_feature/{}.npy'.format(self.log_path, idx), vis_dic)

        # for occupancy metric
        if self.opt.evl_score and save_image:
            occ_metric = {}

            if self.opt.val_depth:
                occ_metric['no_group'] = np.array([
                    # depth
                    np.around(np.array(total_abs_rel_26), 4),
                    np.around(np.array(total_sq_rel_26), 4),
                    np.around(np.array(total_rmse_26), 4),
                    np.around(np.array(total_rmse_log_26), 4),
                    np.around(np.array(total_a1_26), 4),
                    np.around(np.array(total_a2_26), 4),
                    np.around(np.array(total_a3_26), 4),


                    # depth
                    np.around(np.array(total_abs_rel_52), 4),
                    np.around(np.array(total_sq_rel_52), 4),
                    np.around(np.array(total_rmse_52), 4),
                    np.around(np.array(total_rmse_log_52), 4),
                    np.around(np.array(total_a1_52), 4),
                    np.around(np.array(total_a2_52), 4),
                    np.around(np.array(total_a3_52), 4)])


                occ_metric['metric'] = ['abs_rel_26', 'sq_rel_26', 'rmse_26', 'rmse_log_26', 'a1_26', 'a2_26', 'a3_26',
                                        'abs_rel_52', 'sq_rel_52', 'rmse_52', 'rmse_log_52', 'a1_52', 'a2_52', 'a3_52']


                np.save(os.path.join(self.log_path, 'eval', '{}_occupancy.npy'.format(self.local_rank)), occ_metric)


        for eval_type in eval_types:
            for camera_id in errors[eval_type].keys():
                errors[eval_type][camera_id] = np.array(errors[eval_type][camera_id])

        with open(os.path.join(self.log_path, 'eval', '{}.pkl'.format(self.local_rank)), 'wb') as f:
            pickle.dump(errors, f)

        eps_time = time.time() - total_evl_time

        if self.local_rank == 0:
            self.log_print('median: {}'.format(np.array(ratios_median).mean()))
            self.log_print('mean inference time: {}'.format(np.array(total_time).mean()))
            self.log_print('total evl time: {} h'.format(eps_time / 3600))

        print('finish eval!')

        self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        torch.autograd.set_detect_anomaly(True)
        if self.local_rank == 0:
            print("Training")
        self.set_train()

        if self.local_rank == 0:
            self.log_print_train('self.epoch: {}, lr: {}'.format(self.epoch, self.model_lr_scheduler.get_lr()))


        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            losses["loss"].backward()

            self.model_optimizer.step()
            self.model_optimizer.zero_grad()

            torch.cuda.empty_cache()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            # pdb.set_trace()
            if early_phase or late_phase or (self.epoch == (self.opt.num_epochs - 1)):
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        __p = lambda x: basic.pack_seqdim(x, self.opt.B)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.opt.B)

        self.to_device(inputs)

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features, inputs)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}

        total_gt_loss = 0

        if self.opt.volume_depth:
            scale_weight = [1.0, 0.7, 0.5]
        else:
            scale_weight = [1.0, 1.0, 1.0, 1.0]

        depth_gt = inputs['depth']
        mask = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)
        mask.detach_()

        # pdb.set_trace()
        for scale in self.opt.scales:

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0


            disp = outputs[("disp", scale)]
            gt_loss = self.get_gt_loss(inputs, scale, outputs, disp, depth_gt, mask)

            if self.local_rank == 0 and scale == 0:
                print('{} gt loss:'.format(scale), gt_loss)

            losses["gt_loss/{}".format(scale)] = gt_loss
            total_gt_loss += scale_weight[scale] * gt_loss


        total_loss = total_gt_loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def get_gt_loss(self, inputs, scale, outputs, disp, depth_gt, mask):

        singel_scale_total_loss = 0

        if self.opt.volume_depth:

            if self.opt.l1_voxel != 'No':
                density_center = outputs[('density_center', 0)]
                label_true = torch.ones_like(density_center, requires_grad=False)

                all_empty = outputs[('all_empty', 0)]
                label_false = torch.zeros_like(all_empty, requires_grad=False)

                if 'l1' in self.opt.l1_voxel:
                    surface_loss_true = F.l1_loss(density_center, label_true, size_average=True)
                    surface_loss_false = F.l1_loss(all_empty, label_false, size_average=True)

                    total_grid_loss = self.opt.empty_w * surface_loss_false + surface_loss_true

                elif 'ce' in self.opt.l1_voxel:
                    label = torch.cat((label_true, label_false))
                    pred = torch.cat((density_center, all_empty))

                    total_grid_loss = self.criterion(pred, label)

                    if self.local_rank == 0 and scale == 0:
                        print('ce loss:', total_grid_loss)

                else:
                    print('please define the l1_voxel loss')
                    exit()

                if 'only' in self.opt.l1_voxel:
                    depth_pred = disp
                    depth_pred = F.interpolate(depth_pred, size=[self.opt.height_ori, self.opt.width_ori],
                                               mode="bilinear", align_corners=False).squeeze(1)

                    if self.local_rank == 0:
                        print('l1 voxel volume_depth scale: {} , min: {} , max {}:'.format(scale,
                                                                                           depth_pred[0, ...].min(),
                                                                                           depth_pred[0, ...].max()))

                    return total_grid_loss

                else:
                    singel_scale_total_loss += self.opt.surfaceloss * total_grid_loss


            depth_pred = disp
            depth_pred = F.interpolate(depth_pred, size=[self.opt.height_ori, self.opt.width_ori], mode="bilinear", align_corners=False).squeeze(1)

            if self.local_rank == 0:
                print('volume_depth .min, max()', scale, depth_pred[0, ...].min(), depth_pred[0, ...].max())

            no_aug_loss = self.silog_criterion.forward(depth_pred, depth_gt, mask.to(torch.bool), self.opt)
            singel_scale_total_loss += no_aug_loss

            if self.local_rank == 0 and scale == 0:
                print('no_aug_loss：', no_aug_loss)


            if self.opt.disparity_smoothness > 0:
                color = inputs[("color", 0, 0)]
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                # for volume rendering depth , all the depth with 1/4 resolution of the input image
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)
                smoothness_loss = self.opt.disparity_smoothness * smooth_loss
                singel_scale_total_loss += smoothness_loss

        return singel_scale_total_loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, self.opt.max_depth)

        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=self.opt.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        if self.local_rank == 0:
            samples_per_sec = self.opt.batch_size / duration
            time_sofar = time.time() - self.start_time
            training_time_left = (
                                         self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                           " | loss: {:.5f} | time elapsed: {} | time left: {}"

            self.log_print_train(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                               sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        os.makedirs(os.path.join(self.log_path, "eval"), exist_ok=True)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        if self.local_rank == 0:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.step))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                to_save = model.module.state_dict()
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save['height'] = self.opt.height
                    to_save['width'] = self.opt.width
                    to_save['use_stereo'] = self.opt.use_stereo
                torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("adam"))
            torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        if self.local_rank == 0:
            assert os.path.isdir(self.opt.load_weights_folder), \
                "Cannot find folder {}".format(self.opt.load_weights_folder)
            self.log_print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:

            if self.local_rank == 0:
                self.log_print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_optimizer(self):
        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            if self.local_rank == 0:
                self.log_print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            self.log_print("Cannot find Adam weights so Adam is randomly initialized")

    def log_print(self, str):
        print(str)
        with open(os.path.join(self.log_path, 'log.txt'), 'a') as f:
            f.writelines(str + '\n')


    def log_print_train(self, str):
        print(str)
        with open(os.path.join(self.log_path, 'log_train.txt'), 'a') as f:
            f.writelines(str + '\n')


if __name__ == "__main__":
    pass