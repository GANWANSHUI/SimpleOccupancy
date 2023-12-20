#!/bin/bash

cd /home/wsgan/project/bev/SimpleOccupancy



# ddad supervised learning

# python -m torch.distributed.launch --nproc_per_node 8 run.py --config configs/ddad_volume.txt




# nuscene supervised learning

# python -m torch.distributed.launch --nproc_per_node 1 run.py --config configs/nusc_volume.txt



# nuscene self-supervised learning

python -m torch.distributed.launch --nproc_per_node 8 run.py --config configs/nusc_volume_self.txt



# sh /home/wsgan/project/bev/SimpleOccupancy/run.sh
