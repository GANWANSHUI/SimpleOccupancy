# SimpleOccupancy

### [Paper](https://arxiv.org/pdf/2303.10076.pdf)
[//]: # (<br/>)

> A **Simple** Attempt for 3D **Occupancy** Estimation in Autonomous Driving
> 
> Wanshui Gan, Ningkai Mo, Hongbin Xu, Naoto Yokoya 
 
## News

- [2023/4/05]: Update the paper with supplementary material. Code repository is still under construction.
- [2023/3/18]: Initial release.

## Demo  

### Sparse occupancy prediction:

<p align='center'>
<img src="./assets/sparse_demo.gif" width="720px">
<img src="./assets/label.jpg" width="600px">
</p>

### Dense occupancy prediction:
<p align='center'>
<img src="./assets/dense_demo.gif" width="720px">
</p>


## Abstract
The task of estimating 3D occupancy from surrounding-view images is an exciting development in the field of autonomous driving, following the success of Bird's Eye View (BEV) perception. This task provides crucial 3D attributes of the driving environment, enhancing the overall understanding and perception of the surrounding space. However, there is still a lack of a baseline to define the task, such as network design, optimization, and evaluation. In this work, we present a simple attempt for 3D occupancy estimation, which is a CNN-based framework designed to reveal several key factors for 3D occupancy estimation. In addition, we explore the relationship between 3D occupancy estimation and other related tasks, such as monocular depth estimation, stereo matching, and BEV perception (3D object detection and map segmentation), which could advance the study on 3D occupancy estimation. For evaluation, we propose a simple sampling strategy to define the metric for occupancy evaluation, which is flexible for current public datasets. Moreover, we establish a new benchmark in terms of the depth estimation metric, where we compare our proposed method with monocular depth estimation methods on the DDAD and Nuscenes datasets.

## Method 

Proposed network:

<p align='center'>
<img src="./assets/network.png" width="720px">
</p>

Occupancy label and metric comparison:


<p align='center'>
<img src="./assets/metric.png" width="720px">
</p>


## Acknowledgement
Many thanks to these excellent projects:
- [simple_bev](https://github.com/aharley/simple_bev)
- [SurroundDepth](https://github.com/weiyithu/SurroundDepth)


Related Projects:
- [TPVFormer](https://github.com/wzzheng/TPVFormer)
- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- [VoxFormer](https://github.com/NVlabs/VoxFormer)
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)


## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{gan2023simple,
  title={A Simple Attempt for 3D Occupancy Estimation in Autonomous Driving},
  author={Gan, Wanshui and Mo, Ningkai and Xu, Hongbin and Yokoya, Naoto},
  journal={arXiv preprint arXiv:2303.10076},
  year={2023}
}

```




