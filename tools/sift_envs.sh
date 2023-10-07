#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g1
#$ -ac d=nvcr-pytorch-2303
#$ -N sift

# gtn-container_g8.24h

. /fefs/opt/dgx/env_set/nvcr-pytorch-2303.sh

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL


mkdir -p ~/.raiden/nvcr-pytorch-2303
export PATH="${HOME}/.raiden/nvcr-pytorch-2303/bin:$PATH"
export LD_LIBRARY_PATH="${HOME}/.raiden/nvcr-pytorch-2303/lib:$LD_LIBRARY_PATH"
export LDFLAGS=-L/usr/local/nvidia/lib64
export PYTHONPATH="${HOME}/.raiden/nvcr-pytorch-2303/lib/python3.6/site-packages"
export PYTHONUSERBASE="${HOME}/.raiden/nvcr-pytorch-2303"
export PREFIX="${HOME}/.raiden/nvcr-pytorch-2303"


# cp -r /home/wsgan/.raiden/aip-pytorch-2105-gl1mesa-1 /home/wsgan

# qrsh -jc gtn-container_g1_dev.default -ac d=nvcr-pytorch-2303

# pip install opencv-python-headless==3.4.16.59 opencv-contrib-python-headless==3.4.16.59

# pip install opencv-python-headless==3.4.18.65 opencv-contrib-python-headless==3.4.18.65

# qsub /home/wsgan/project/bev/S3DO/tools/sift_envs.sh

# sh /home/wsgan/project/bev/S3DO/tools/sift_envs.sh

cd /home/wsgan/project/bev/S3DO/tools

# python sift_ddad.py

# python match_ddad.py

# python sift_nusc.py

python match_nusc.py
