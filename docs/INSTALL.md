<img src="../figs/logo.png" align="right" width="20%">

# Installation

### General Requirements

This codebase is tested with `torch==1.11.0` and `torchvision==0.12.0`, with `CUDA 11.3`. In order to successfully reproduce the results reported in our paper, we recommend you to follow the exact same configuation with us. However, similar versions that came out lately should be good as well.

### Range View

For the **range view option**, we use [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI) as the LiDAR segmentation backbone. We adopt its *ResNet34-point* variant as recommended in the [original paper](https://arxiv.org/abs/2109.03787), which contains `6.05M` parameters. The resolutions of the rasterized range image are set as `32x1920` for nuScenes and `64x1920` for SemanticKITTI and ScribbleKITTI.


### Voxel
For the **voxel option**, we use a more compact version of [Cylinder3D](https://github.com/xinge008/Cylinder3D) as the LiDAR segmentation backbone. It contains `28.13M` parameters (compared to 56.26M for the one used in the [original paper](https://arxiv.org/abs/2011.10033)). We also use a smaller voxel resolution `[240, 180, 32]` compared to the original configuration `[480, 360, 32]`. This saves around 4x memory consumption and further helps to speed up training.

#### Requirements
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [spconv](https://github.com/traveller59/spconv/tree/v1.2.1) (v1.2.1)
- [pybind11](https://github.com/pybind/pybind11/tree/085a29436a8c472caaaf7157aa644b571079bcaa)


### Step 1: Create Enviroment
```
conda create -n lasermix python=3.10
```

### Step 2: Activate Enviroment
```
conda activate lasermix
```

### Step 3: Install PyTorch
```
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

### Step 4: Install Necessary Libraries
#### 4.1 - [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) :oncoming_automobile:
```
pip install nuscenes-devkit 
```

#### 4.2 - [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)
```
conda install pytorch-scatter -c pyg
```

#### 4.3 - [SparseConv](https://github.com/traveller59/spconv)
```
pip install spconv_cu113
```

#### 4.4 - Other Packages
```
pip install pyyaml easydict numba
```



