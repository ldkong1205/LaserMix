<img src="../figs/logo.png" align="right" width="20%">

# Installation

### General Requirements

Coming soon.

### Range View

Coming soon.


### Voxel
For the **voxel option**, we use a more compact version of [Cylinder3D](https://github.com/xinge008/Cylinder3D) as the segmentation backbone. It contains 28.13M parameters (compared to 56.26M for the one used in the [original paper](https://arxiv.org/abs/2011.10033)). We also use a smaller voxel resolution [240, 180, 20] compared to the original configuration [480, 360, 32]. This saves around 4x memory consumption and further helps to speed up training.

#### Requirements
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [spconv](https://github.com/traveller59/spconv/tree/v1.2.1) (v1.2.1)
- [pybind11](https://github.com/pybind/pybind11/tree/085a29436a8c472caaaf7157aa644b571079bcaa)
