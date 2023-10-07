## Environment Preparation
Note that the code base of [SimpleOccupancy](https://github.com/GANWANSHUI/SimpleOccupancy) is adapted from [Surrounddepth](https://github.com/weiyithu/SurroundDepth), please refer to it if something is unclear.
One more thing we need to do is to get the pointcloud under the ego coordinate. 

* python 3.8, pytorch 1.12.1, CUDA 11.4, A 100
```bash
git clone https://github.com/GANWANSHUI/SimpleOccupancy.git
conda create -n SimpleOccupancy python=3.8
conda activate SimpleOccupancy
pip install -r requirements.txt
```
Since we use [dgp codebase](https://github.com/TRI-ML/dgp) to generate groundtruth depth, you should also install it. 

## Data Preparation
Datasets are assumed to be downloaded under `data/<dataset-name>`.
In general, we use the data structure the same as [Surroundepth](https://github.com/weiyithu/SurroundDepth). For the supervision training, we need to export the depth map and the point cloud for both the training and testing sets.


### DDAD
* Please download the official [DDAD dataset](https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar) and place them under `data/ddad/raw_data`. You may refer to official [DDAD repository](https://github.com/TRI-ML/DDAD) for more info and instructions.
* Please download [metadata](https://cloud.tsinghua.edu.cn/f/50cb1ea5b1344db8b51c/?dl=1) of DDAD and place these pkl files in `datasets/ddad`.
* Export depth maps for evaluation 
```bash
cd tools
# for depth map
python export_gt_depth_ddad.py val path_to_ddad_raw_data
python export_gt_depth_ddad.py train path_to_ddad_raw_data

# for point cloud
python export_point_cloud_ddad.py train path_to_ddad_raw_data

```

* The final data structure should be:
```
SurroundDepth
├── data
│   ├── ddad
│   │   │── raw_data
│   │   │   │── 000000
|   |   |   |── ...
|   |   |── depth
│   │   │   │── 000000
|   |   |   |── ...
|   |   |── point_cloud
│   │   │   │── 000000
|   |   |   |── ...
|   |   |── mask
│   │   │   │── 000000
|   |   |   |── ...
```

### nuScenes
* Please download the official [nuScenes dataset](https://www.nuscenes.org/download) to `data/nuscenes/raw_data`
* Export depth maps for evaluation 
```bash
cd tools
python export_gt_depth_nusc.py val

```
* The final data structure should be:
```
SurroundDepth
├── data
│   ├── nuscenes
│   │   │── raw_data
│   │   │   │── samples
|   |   |   |── sweeps
|   |   |   |── maps
|   |   |   |── v1.0-trainval
|   |   |── depth
│   │   │   │── samples
|   |   |── match
│   │   │   │── samples
```
