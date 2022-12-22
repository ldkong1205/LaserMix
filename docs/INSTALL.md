<img src="../docs/figs/logo.png" align="right" width="20%">

# Installation

### General Requirements

This codebase is tested with `torch==1.11.0` and `torchvision==0.12.0`, with `CUDA 11.3`. In order to successfully reproduce the results reported in our paper, we recommend you to follow the exact same configuation with us. However, similar versions that came out lately should be good as well.

### Range View

For the **range view option**, we use [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI) as the LiDAR segmentation backbone. We adopt its *ResNet34-point* variant as recommended in the [original paper](https://arxiv.org/abs/2109.03787), which contains `6.05M` parameters. The resolutions of the rasterized range image are set as `32x1920` for nuScenes and `64x2048` for SemanticKITTI and ScribbleKITTI.

:memo: **Updated 2022.12.21:** We now support three more mainstream range view LiDAR segmentation backbones, including [RangeNet++](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf), [SalsaNext](https://arxiv.org/abs/2003.03653), and [CENet](https://arxiv.org/abs/2207.12691), with horizontal rasterization resolutions of `512`, `960`, `1024`, `1920`, and `2048`.


### Voxel
For the **voxel option**, we use a more compact version of [Cylinder3D](https://github.com/xinge008/Cylinder3D) as the LiDAR segmentation backbone. It contains `28.13M` parameters (compared to 56.26M for the one used in the [original paper](https://arxiv.org/abs/2011.10033)). We also use a smaller voxel resolution `[240, 180, 32]` compared to the original configuration `[480, 360, 32]`. This saves around 4x memory consumption and further helps to speed up training.

:memo: **Updated 2022.12.21:** We now support two more mainstream voxel-based LiDAR segmentation backbones, i.e., [MinkowskiUNet](https://github.com/NVIDIA/MinkowskiEngine) and [SPVCNN](https://arxiv.org/pdf/2007.16100). We also enable the use of both `cylinder` and `cubic` voxelizations, with various voxel lengths, under faster and more efficient sparse convolutional operations.

#### Requirements
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [spconv](https://github.com/traveller59/spconv/tree/v1.2.1) (cu113)
- [torchsparse](https://github.com/mit-han-lab/torchsparse)


### Step 1: Create Enviroment
```Shell
conda create -n lasermix python=3.10
```

### Step 2: Activate Enviroment
```Shell
conda activate lasermix
```

### Step 3: Install PyTorch
```Shell
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

### Step 4: Install Necessary Libraries
#### 4.1 - [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
:oncoming_automobile: **Note:** This toolkit is **required** in order to run experiments on the [nuScenes](https://www.nuscenes.org/nuscenes) dataset.
```Shell
pip install nuscenes-devkit 
```

#### 4.2 - [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)
```Shell
conda install pytorch-scatter -c pyg
```

#### 4.3 - [SparseConv](https://github.com/traveller59/spconv)
```Shell
pip install spconv_cu113
```

#### 4.4 - [TorchSparse](https://github.com/mit-han-lab/torchsparse)
:cactus: **Note:** The following steps are **required** in order to use the `voxel` backbones in this codebase.

- Make a directory named `torchsparse_dir`
```Shell
cd package/
mkdir torchsparse_dir/
```

- Unzip the `.zip` files in `package/`
```Shell
unzip sparsehash.zip
unzip torchsparse.zip

mv sparsehash-master/ sparsehash/
mv torchsparse-master/ torchsparse/
```

- Setup `sparsehash` (Note that `${ROOT}` should be your home path to the LaserMix folder)
```Shell
cd sparsehash/
./configure --prefix=/${ROOT}/LaserMix/package/torchsparse_dir/SparsehasH/
```
```Shell
make
```
```Shell
make install
```

- Compile `torchsparse`
```Shell
cd ..
pip install ./torchsparse
```

- It takes a while to build wheels. After successfully building `torchsparse`, you should see the following:
```Shell
Successfully built torchsparse
Installing collected packages: torchsparse
Successfully installed torchsparse-2.0.0b0
```



#### 4.5 - Other Packages
```Shell
pip install pyyaml easydict numba
```



