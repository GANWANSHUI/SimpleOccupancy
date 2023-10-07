import os
import shutil
import pdb

g = os.walk(r"/home/wsgan/project/bev/SurroundDepth/data/nuscenes")

for path,dir_list,file_list in g:
    for file_name in file_list:

        target_path = path.replace('/home/wsgan/project/bev/SurroundDepth', '/data/ggeoinfo/Wanshui_BEV')

        source_path = os.path.join(path, file_name)

        target_name = os.path.join(target_path, file_name)

        if os.path.exists(target_name):
            print('{} exist'.format(target_name))
            pass
        else:
            os.makedirs(target_path, exist_ok=True)
            # print('makeing')
            shutil.copy(source_path, target_path)
            print('{} copying'.format(source_path))


        # print(target_path)
        # pdb.set_trace()

        # if '812' in file_name:
        #
        #     print(os.path.join(path, file_name))
        #     # os.remove(os.path.join(path, file_name))


# python /home/wsgan/project/bev/SurroundDepth/datasets/data_movement.py