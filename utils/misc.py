import pdb
import numpy as np
import torch
import os, cv2

class SimplePool():
    def __init__(self, pool_size, version='pt'):
        self.pool_size = pool_size
        self.version = version
        # random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.items = []
        if not (version=='pt' or version=='np'):
            print('version = %s; please choose pt or np')
            assert(False) # please choose pt or np
            
    def __len__(self):
        return len(self.items)
    
    def mean(self, min_size='none'):
        if min_size=='half':
            pool_size_thresh = self.pool_size/2
        else:
            pool_size_thresh = 1
            
        if self.version=='np':
            if len(self.items) >= pool_size_thresh:
                return np.sum(self.items)/len(self.items)
            else:
                return np.nan
        if self.version=='pt':
            if len(self.items) >= pool_size_thresh:
                return torch.sum(self.items)/len(self.items)
            else:
                return torch.from_numpy(np.nan)
    
    def sample(self):
        idx = np.random.randint(len(self.items))
        return self.items[idx]
    
    def fetch(self, num=None):
        if self.version=='pt':
            item_array = torch.stack(self.items)
        elif self.version=='np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert(len(self.items) >= num)
                
            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
    
    def empty(self):
        self.items = []
        self.num = 0
            
    def update(self, items):
        for item in items:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.items.pop(0)
            # add to the back
            self.items.append(item)
        return self.items



def get_video(file_path =None):
    file_path = '/home/wsgan/project/bev/SurroundDepth/logs/ddad_ablation_50_6m/1218_1204_vis_1_all_epoch_12/T_No_val_0.4_center_True_voxel_l1_N_30_sur_5.0_empty_w_2.0_depth_0.1_52.0_loss_silog_out_1_en_101_actfun_Softplus_input_64_16_vtrans_simple/log_False_step_0.5_alpha_0.01_sm_2.0_en_0.0001_de_0.001_volume_True_t2v_interpolation_aggregation_3dcnn_loss_gt_tv_0.05_type_prob_pe_embedding/visual_new'
    # file_path = os.path.join('\\\?\\' + file_path)
    # file_list = os.listdir(file_path)
    # pdb.set_trace()
    #
    # file_list = [os.path.join(file_path, i) for i in file_list]


    path_list = os.listdir(file_path)
    path_list.sort(key=lambda x: int(x[:-4]))  # 将'.jpg'左边的字符转换成整数型进行排序
    path_list = [os.path.join(file_path, i) for i in path_list]
    print(path_list)

    video = cv2.VideoWriter('{}/0.avi'.format(file_path), cv2.VideoWriter_fourcc(*'MJPG'), 15, (2016, 1344))  # 定义保存视频目录名称及压缩格式，fps=10,像素为1280*720

    for path in path_list:
        img = cv2.imread(path)  # 读取图片
        # img = cv2.resize(img, (1280, 720))  # 将图片转换为1280*720
        video.write(img)  # 写入视频


if __name__ == '__main__':
    get_video()