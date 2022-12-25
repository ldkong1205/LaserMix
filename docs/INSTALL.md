<img src="../docs/figs/logo.png" align="right" width="20%">

# Installation

### General Requirements

This codebase is tested with `torch==1.11.0` and `torchvision==0.12.0`, with `CUDA 11.3`. In order to successfully reproduce the results reported in our paper, we recommend you to follow the exact same configuation with us. However, similar versions that came out lately should be good as well.

### Range View

- For the **range view option**, we use [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI) as the LiDAR segmentation backbone. We adopt its *ResNet34-point* variant as recommended in the [original paper](https://arxiv.org/abs/2109.03787), which contains `6.05M` parameters. The resolutions of the rasterized range image are set as `32x1920` for nuScenes and `64x2048` for SemanticKITTI and ScribbleKITTI.

- :memo: **Updated [2022.12]:** We now support three more mainstream range view LiDAR segmentation backbones, including [RangeNet++](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf), [SalsaNext](https://arxiv.org/abs/2003.03653), and [CENet](https://arxiv.org/abs/2207.12691), with horizontal rasterization resolutions of `512`, `960`, `1024`, `1920`, and `2048`.


### Voxel
- For the **voxel option**, we use a more compact version of [Cylinder3D](https://github.com/xinge008/Cylinder3D) as the LiDAR segmentation backbone. It contains `28.13M` parameters (compared to 56.26M for the one used in the [original paper](https://arxiv.org/abs/2011.10033)). We also use a smaller voxel resolution `[240, 180, 32]` compared to the original configuration `[480, 360, 32]`. This saves around 4x memory consumption and further helps to speed up training.

- :memo: **Updated [2022.12]:** We now support two more mainstream voxel-based LiDAR segmentation backbones, i.e., [MinkowskiUNet](https://github.com/NVIDIA/MinkowskiEngine) and [SPVCNN](https://arxiv.org/pdf/2007.16100). We also enable the use of both `cylinder` and `cubic` voxelizations, with various voxel lengths, under faster and more efficient sparse convolutional operations.


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

- Setup `sparsehash` (Note that `${ROOT}` should be your home path to the `LaserMix` folder)
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


## Enviroment Summary

We provide the list of all packages and their corresponding versions installed in this codebase:
```Shell
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
anyio                     3.6.2                    pypi_0    pypi
argon2-cffi               21.3.0                   pypi_0    pypi
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
arrow                     1.2.3                    pypi_0    pypi
asttokens                 2.2.1                    pypi_0    pypi
attrs                     22.2.0                   pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
beautifulsoup4            4.11.1                   pypi_0    pypi
blas                      1.0                         mkl  
bleach                    5.0.1                    pypi_0    pypi
brotlipy                  0.7.0           py310h7f8727e_1002  
bzip2                     1.0.8                h7b6447c_0  
ca-certificates           2022.10.11           h06a4308_0  
cachetools                5.2.0                    pypi_0    pypi
ccimport                  0.4.2                    pypi_0    pypi
certifi                   2022.12.7       py310h06a4308_0  
cffi                      1.15.1          py310h5eee18b_3  
charset-normalizer        2.0.4              pyhd3eb1b0_0  
comm                      0.1.2                    pypi_0    pypi
contourpy                 1.0.6                    pypi_0    pypi
cryptography              38.0.1          py310h9ce1e76_0  
cudatoolkit               11.3.1               h2bc3f7f_2  
cumm-cu113                0.3.7                    pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
debugpy                   1.6.4                    pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
defusedxml                0.7.1                    pypi_0    pypi
descartes                 1.1.0                    pypi_0    pypi
easydict                  1.10                     pypi_0    pypi
entrypoints               0.4                      pypi_0    pypi
executing                 1.2.0                    pypi_0    pypi
fastjsonschema            2.16.2                   pypi_0    pypi
ffmpeg                    4.3                  hf484d3e_0    pytorch
fire                      0.5.0                    pypi_0    pypi
flit-core                 3.6.0              pyhd3eb1b0_0  
fonttools                 4.38.0                   pypi_0    pypi
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.12.1               h4a9f257_0  
giflib                    5.2.1                h7b6447c_0  
gmp                       6.2.1                h295c915_3  
gnutls                    3.6.15               he1e5248_0  
idna                      3.4             py310h06a4308_0  
intel-openmp              2021.4.0          h06a4308_3561  
ipykernel                 6.19.4                   pypi_0    pypi
ipython                   8.7.0                    pypi_0    pypi
ipython-genutils          0.2.0                    pypi_0    pypi
ipywidgets                8.0.4                    pypi_0    pypi
isoduration               20.11.0                  pypi_0    pypi
jedi                      0.18.2                   pypi_0    pypi
jinja2                    3.1.2                    pypi_0    pypi
joblib                    1.2.0                    pypi_0    pypi
jpeg                      9e                   h7f8727e_0  
jsonpointer               2.3                      pypi_0    pypi
jsonschema                4.17.3                   pypi_0    pypi
jupyter                   1.0.0                    pypi_0    pypi
jupyter-client            7.4.8                    pypi_0    pypi
jupyter-console           6.4.4                    pypi_0    pypi
jupyter-core              5.1.1                    pypi_0    pypi
jupyter-events            0.5.0                    pypi_0    pypi
jupyter-server            2.0.5                    pypi_0    pypi
jupyter-server-terminals  0.4.3                    pypi_0    pypi
jupyterlab-pygments       0.2.2                    pypi_0    pypi
jupyterlab-widgets        3.0.5                    pypi_0    pypi
kiwisolver                1.4.4                    pypi_0    pypi
lame                      3.100                h7b6447c_0  
lark                      1.1.5                    pypi_0    pypi
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libdeflate                1.8                  h7f8727e_5  
libffi                    3.4.2                h6a678d5_6  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libiconv                  1.16                 h7f8727e_2  
libidn2                   2.3.2                h7f8727e_0  
libpng                    1.6.37               hbc83047_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtasn1                  4.16.0               h27cfd23_0  
libtiff                   4.4.0                hecacb30_2  
libunistring              0.9.10               h27cfd23_0  
libuuid                   1.41.5               h5eee18b_0  
libuv                     1.40.0               h7b6447c_0  
libwebp                   1.2.4                h11a3e52_0  
libwebp-base              1.2.4                h5eee18b_0  
llvmlite                  0.39.1                   pypi_0    pypi
lz4-c                     1.9.4                h6a678d5_0  
markupsafe                2.1.1                    pypi_0    pypi
matplotlib                3.6.2                    pypi_0    pypi
matplotlib-inline         0.1.6                    pypi_0    pypi
mistune                   2.0.4                    pypi_0    pypi
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0           py310h7f8727e_0  
mkl_fft                   1.3.1           py310hd6ae3a3_0  
mkl_random                1.2.2           py310h00e6091_0  
nbclassic                 0.4.8                    pypi_0    pypi
nbclient                  0.7.2                    pypi_0    pypi
nbconvert                 7.2.7                    pypi_0    pypi
nbformat                  5.7.1                    pypi_0    pypi
ncurses                   6.3                  h5eee18b_3  
nest-asyncio              1.5.6                    pypi_0    pypi
nettle                    3.7.3                hbbd107a_1  
ninja                     1.11.1                   pypi_0    pypi
notebook                  6.5.2                    pypi_0    pypi
notebook-shim             0.2.2                    pypi_0    pypi
numba                     0.56.4                   pypi_0    pypi
numpy                     1.23.4          py310hd5efca6_0  
numpy-base                1.23.4          py310h8e6c178_0  
nuscenes-devkit           1.1.9                    pypi_0    pypi
opencv-python             4.6.0.66                 pypi_0    pypi
openh264                  2.1.1                h4ff587b_0  
openssl                   1.1.1s               h7f8727e_0  
packaging                 22.0                     pypi_0    pypi
pandocfilters             1.5.0                    pypi_0    pypi
parso                     0.8.3                    pypi_0    pypi
pccm                      0.4.4                    pypi_0    pypi
pexpect                   4.8.0                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    9.3.0           py310hace64e9_1  
pip                       22.3.1          py310h06a4308_0  
platformdirs              2.6.0                    pypi_0    pypi
portalocker               2.6.0                    pypi_0    pypi
prometheus-client         0.15.0                   pypi_0    pypi
prompt-toolkit            3.0.36                   pypi_0    pypi
psutil                    5.9.4                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.2                    pypi_0    pypi
pybind11                  2.10.2                   pypi_0    pypi
pycocotools               2.0.6                    pypi_0    pypi
pycparser                 2.21               pyhd3eb1b0_0  
pygments                  2.13.0                   pypi_0    pypi
pyopenssl                 22.0.0             pyhd3eb1b0_0  
pyparsing                 3.0.9                    pypi_0    pypi
pyquaternion              0.9.9                    pypi_0    pypi
pyrsistent                0.19.2                   pypi_0    pypi
pysocks                   1.7.1           py310h06a4308_0  
python                    3.10.8               h7a1cb2a_1  
python-dateutil           2.8.2                    pypi_0    pypi
python-json-logger        2.0.4                    pypi_0    pypi
pytorch                   1.11.0          py3.10_cuda11.3_cudnn8.2.0_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytorch-scatter           2.0.9           py310_torch_1.11.0_cu113    pyg
pyyaml                    6.0                      pypi_0    pypi
pyzmq                     24.0.1                   pypi_0    pypi
qtconsole                 5.4.0                    pypi_0    pypi
qtpy                      2.3.0                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
requests                  2.28.1          py310h06a4308_0  
rfc3339-validator         0.1.4                    pypi_0    pypi
rfc3986-validator         0.1.1                    pypi_0    pypi
scikit-learn              1.2.0                    pypi_0    pypi
scipy                     1.9.3                    pypi_0    pypi
send2trash                1.8.0                    pypi_0    pypi
setuptools                65.5.0          py310h06a4308_0  
shapely                   2.0.0                    pypi_0    pypi
six                       1.16.0             pyhd3eb1b0_1  
sniffio                   1.3.0                    pypi_0    pypi
soupsieve                 2.3.2.post1              pypi_0    pypi
spconv-cu113              2.2.6                    pypi_0    pypi
sqlite                    3.40.0               h5082296_0  
stack-data                0.6.2                    pypi_0    pypi
termcolor                 2.1.1                    pypi_0    pypi
terminado                 0.17.1                   pypi_0    pypi
threadpoolctl             3.1.0                    pypi_0    pypi
tinycss2                  1.2.1                    pypi_0    pypi
tk                        8.6.12               h1ccaba5_0  
torchsparse               2.0.0b0                  pypi_0    pypi
torchvision               0.12.0              py310_cu113    pytorch
tornado                   6.2                      pypi_0    pypi
tqdm                      4.64.1                   pypi_0    pypi
traitlets                 5.8.0                    pypi_0    pypi
typing_extensions         4.4.0           py310h06a4308_0  
tzdata                    2022g                h04d1e81_0  
uri-template              1.2.0                    pypi_0    pypi
urllib3                   1.26.13         py310h06a4308_0  
wcwidth                   0.2.5                    pypi_0    pypi
webcolors                 1.12                     pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.4.2                    pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0  
widgetsnbextension        4.0.5                    pypi_0    pypi
xz                        5.2.8                h5eee18b_0  
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.2                ha4553b6_0 
```

