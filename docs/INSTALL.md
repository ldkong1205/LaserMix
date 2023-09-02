<img src="../docs/figs/logo.png" align="right" width="20%">

# Installation

### General Requirement

This codebase is tested with `torch==1.10.0`, `torchvision==0.11.0`, `mmcv==2.0.0rc4`, `mmdet3d==1.2.0`, and `mmengine==0.8.4`, with `CUDA 11.3`. In order to successfully reproduce the results reported in our paper, we recommend you follow the exact same configuration with us. However, similar versions that came out lately should be good as well.

### Range View

- For the **range view option**, we use [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI) as the LiDAR segmentation backbone. We adopt its *ResNet34-point* variant as recommended in the [original paper](https://arxiv.org/abs/2109.03787), which contains `6.05M` parameters. The resolutions of the rasterized range image are set as `32x1920` for nuScenes and `64x2048` for SemanticKITTI and ScribbleKITTI.

- :memo: **Updated [2022.12]:** We now support three more mainstream range view LiDAR segmentation backbones, including [RangeNet++](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf), [SalsaNext](https://arxiv.org/abs/2003.03653), and [CENet](https://arxiv.org/abs/2207.12691), with horizontal rasterization resolutions of `512`, `960`, `1024`, `1920`, and `2048`.


### Cylinder
- For the **cylinder option**, we use a more compact version of [Cylinder3D](https://github.com/xinge008/Cylinder3D) as the LiDAR segmentation backbone. It contains `28.13M` parameters (compared to 56.26M for the one used in the [paper](https://arxiv.org/abs/2011.10033)). We also use a smaller voxel resolution `[240, 180, 32]` compared to the original configuration `[480, 360, 32]`. This saves around 4x memory consumption and further helps to speed up training.


### Voxel
- For the **voxel option**, We support two mainstream voxel-based LiDAR segmentation backbones, i.e., [MinkowskiUNet](https://github.com/NVIDIA/MinkowskiEngine) and [SPVCNN](https://arxiv.org/pdf/2007.16100). We also enable using both `cylinder` and `cubic` voxelizations, with various voxel lengths, under faster and more efficient sparse convolutional operations.

<hr>

### Step 1: Create Environment
```Shell
conda create -n lasermix python=3.10
```

### Step 2: Activate Environment
```Shell
conda activate lasermix
```

### Step 3: Install PyTorch
```Shell
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

### Step 4: Install MMDetection3D
- **Step 4.1:** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv), and [MMDetection](https://github.com/open-mmlab/mmdetection) using [MIM](https://github.com/open-mmlab/mim)
  ```Shell
  pip install -U openmim
  mim install mmengine
  mim install 'mmcv>=2.0.0rc4'
  mim install 'mmdet>=3.0.0'
  ```
  **Note:** In MMCV-v2.x, `mmcv-full` is renamed to `mmcv`, if you want to install `mmcv` without CUDA ops, you can use `mim install "mmcv-lite>=2.0.0rc4"` to install the lite version.

- **Step 4.2:** Install MMDetection3D
  - **Option One:** If you develop and run `mmdet3d` directly, install it from the source:
    ```Shell
    git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
    ```
    **Note:** `"-b dev-1.x"` means checkout to the `dev-1.x` branch.
    
    ```Shell
    cd mmdetection3d
    pip install -v -e .
    ```
    **Note:** `"-v"` means verbose, or more output, `"-e"` means installing a project in editable mode, thus any local modifications made to the code will take effect without reinstallation.

  - **Option Two:** If you use `mmdet3d` as a dependency or third-party package, install it with [MIM](https://github.com/open-mmlab/mim):
    ```Shell
    mim install "mmdet3d>=1.1.0"
    ```

### Step 5: Install Sparse Convolution Backend

- **Step 5.1:** Install SPConv
  - We have supported `spconv 2.0`. If the user has installed `spconv 2.0`, the code will use `spconv 2.0` by default, which will take up less GPU memory than using the default `mmcv` version `spconv`. Users can use the following commands to install `spconv 2.0`:
    ```Shell
    pip install cumm-cuxxx
    pip install spconv-cuxxx
    ```
    Where `xxx` is the CUDA version in the environment. For example, using CUDA 11.3, the command will be `pip install cumm-cu113 && pip install spconv-cu113`.
  
  - The supported CUDA versions include `10.2`, `11.1`, `11.3`, and `11.4`. Users can also install it by building from the source. For more details please refer to [spconv v2.x](https://github.com/traveller59/spconv).


- **Step 5.2:** Install TorchSparse
  - If necessary, follow the [original installation guide](https://github.com/mit-han-lab/torchsparse#installation) or use pip to install it:
    ```Shell
    sudo apt-get install libsparsehash-dev
    pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    ```
    
  - Or omit sudo install by following command:
    ```Shell
    conda install -c bioconda sparsehash
    export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
    # replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
    pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
    ```

- **Step 5.3:** Install Minkowski Engine (Optional)
  - We also support the Minkowski Engine as a sparse convolution backend. If necessary, follow the [original installation guide](https://github.com/NVIDIA/MinkowskiEngine#installation) or use pip to install it:
    ```Shell
    conda install openblas-devel -c anaconda
    export CPLUS_INCLUDE_PATH=CPLUS_INCLUDE_PATH:${YOUR_CONDA_ENVS_DIR}/include
    # replace ${YOUR_CONDA_ENVS_DIR} to your anaconda environment path e.g. `/home/username/anaconda3/envs/openmmlab`.
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
    ```


### Step 6: Install nuScenes Devkit
:oncoming_automobile: The [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) is **required** in order to run experiments on the [nuScenes](https://www.nuscenes.org/nuscenes) dataset.
```Shell
pip install nuscenes-devkit 
```




## Environment Summary

We provide the list of all packages and their corresponding versions installed in this codebase:
```Shell
# Name                    Version                   Build    Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
absl-py                   1.4.0                    pypi_0    pypi
addict                    2.4.0                    pypi_0    pypi
aiofiles                  22.1.0                   pypi_0    pypi
aiosqlite                 0.19.0                   pypi_0    pypi
anyio                     3.7.1                    pypi_0    pypi
argon2-cffi               21.3.0                   pypi_0    pypi
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
arrow                     1.2.3                    pypi_0    pypi
asttokens                 2.2.1                    pypi_0    pypi
attrs                     23.1.0                   pypi_0    pypi
babel                     2.12.1                   pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
beautifulsoup4            4.12.2                   pypi_0    pypi
black                     23.7.0                   pypi_0    pypi
blas                      1.0                         mkl  
bleach                    6.0.0                    pypi_0    pypi
bzip2                     1.0.8                h7f98852_4    conda-forge
ca-certificates           2023.05.30           h06a4308_0  
cachetools                5.3.1                    pypi_0    pypi
certifi                   2023.5.7                 pypi_0    pypi
cffi                      1.15.1                   pypi_0    pypi
charset-normalizer        3.2.0                    pypi_0    pypi
click                     8.1.6                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
comm                      0.1.3                    pypi_0    pypi
contourpy                 1.1.0                    pypi_0    pypi
cudatoolkit               11.3.1              h9edb442_10    conda-forge
cycler                    0.11.0                   pypi_0    pypi
debugpy                   1.6.7                    pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
defusedxml                0.7.1                    pypi_0    pypi
deprecation               2.1.0                    pypi_0    pypi
descartes                 1.1.0                    pypi_0    pypi
exceptiongroup            1.1.2                    pypi_0    pypi
executing                 1.2.0                    pypi_0    pypi
fastjsonschema            2.17.1                   pypi_0    pypi
ffmpeg                    4.3                  hf484d3e_0    pytorch
fire                      0.5.0                    pypi_0    pypi
flake8                    5.0.4                    pypi_0    pypi
fonttools                 4.41.0                   pypi_0    pypi
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.10.4               h0708190_1    conda-forge
giflib                    5.2.1                h5eee18b_3  
gmp                       6.2.1                h58526e2_0    conda-forge
gnutls                    3.6.13               h85f3911_1    conda-forge
google-auth               2.22.0                   pypi_0    pypi
google-auth-oauthlib      1.0.0                    pypi_0    pypi
grpcio                    1.56.0                   pypi_0    pypi
idna                      3.4                      pypi_0    pypi
imageio                   2.31.1                   pypi_0    pypi
importlib-metadata        6.8.0                    pypi_0    pypi
importlib-resources       6.0.0                    pypi_0    pypi
iniconfig                 2.0.0                    pypi_0    pypi
intel-openmp              2021.4.0          h06a4308_3561  
ipykernel                 6.24.0                   pypi_0    pypi
ipython                   8.12.2                   pypi_0    pypi
ipython-genutils          0.2.0                    pypi_0    pypi
ipywidgets                8.0.7                    pypi_0    pypi
isoduration               20.11.0                  pypi_0    pypi
jedi                      0.18.2                   pypi_0    pypi
jinja2                    3.1.2                    pypi_0    pypi
joblib                    1.3.1                    pypi_0    pypi
jpeg                      9e                   h166bdaf_1    conda-forge
json5                     0.9.14                   pypi_0    pypi
jsonpointer               2.4                      pypi_0    pypi
jsonschema                4.18.4                   pypi_0    pypi
jsonschema-specifications 2023.7.1                 pypi_0    pypi
jupyter                   1.0.0                    pypi_0    pypi
jupyter-client            8.3.0                    pypi_0    pypi
jupyter-console           6.6.3                    pypi_0    pypi
jupyter-core              5.3.1                    pypi_0    pypi
jupyter-events            0.6.3                    pypi_0    pypi
jupyter-packaging         0.12.3                   pypi_0    pypi
jupyter-server            2.7.0                    pypi_0    pypi
jupyter-server-fileid     0.9.0                    pypi_0    pypi
jupyter-server-terminals  0.4.4                    pypi_0    pypi
jupyter-server-ydoc       0.8.0                    pypi_0    pypi
jupyter-ydoc              0.2.5                    pypi_0    pypi
jupyterlab                3.6.5                    pypi_0    pypi
jupyterlab-pygments       0.2.2                    pypi_0    pypi
jupyterlab-server         2.23.0                   pypi_0    pypi
jupyterlab-widgets        3.0.8                    pypi_0    pypi
kiwisolver                1.4.4                    pypi_0    pypi
lame                      3.100             h7f98852_1001    conda-forge
lazy-loader               0.3                      pypi_0    pypi
lcms2                     2.12                 hddcbb42_0    conda-forge
libedit                   3.1.20221030         h5eee18b_0  
libffi                    3.2.1             hf484d3e_1007  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libiconv                  1.17                 h166bdaf_0    conda-forge
libpng                    1.6.37               h21135ba_2    conda-forge
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.2.0                hecacb30_2  
libuv                     1.43.0               h7f98852_0    conda-forge
libwebp                   1.2.2                h55f646e_0  
libwebp-base              1.2.2                h7f98852_1    conda-forge
llvmlite                  0.40.1                   pypi_0    pypi
lyft-dataset-sdk          0.0.8                    pypi_0    pypi
lz4-c                     1.9.3                h9c3ff4c_1    conda-forge
markdown                  3.4.3                    pypi_0    pypi
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                2.1.3                    pypi_0    pypi
matplotlib                3.5.2                    pypi_0    pypi
matplotlib-inline         0.1.6                    pypi_0    pypi
mccabe                    0.7.0                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
mistune                   3.0.1                    pypi_0    pypi
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0            py38h95df7f1_0    conda-forge
mkl_fft                   1.3.1            py38h8666266_1    conda-forge
mkl_random                1.2.2            py38h1abd341_0    conda-forge
mmcv                      2.0.0rc4                 pypi_0    pypi
mmdet                     3.0.0                    pypi_0    pypi
mmdet3d                   1.2.0                     dev_0    <develop>
mmengine                  0.8.4                    pypi_0    pypi
model-index               0.1.11                   pypi_0    pypi
mypy-extensions           1.0.0                    pypi_0    pypi
nbclassic                 1.0.0                    pypi_0    pypi
nbclient                  0.8.0                    pypi_0    pypi
nbconvert                 7.7.1                    pypi_0    pypi
nbformat                  5.9.1                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.5.6                    pypi_0    pypi
nettle                    3.6                  he412f7d_0    conda-forge
networkx                  3.1                      pypi_0    pypi
notebook                  6.5.4                    pypi_0    pypi
notebook-shim             0.2.3                    pypi_0    pypi
numba                     0.57.1                   pypi_0    pypi
numpy                     1.24.3           py38h14f4228_0  
numpy-base                1.24.3           py38h31eccc5_0  
nuscenes-devkit           1.1.10                   pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi
olefile                   0.46               pyh9f0ad1d_1    conda-forge
open3d                    0.13.0                   pypi_0    pypi
opencv-python             4.8.0.74                 pypi_0    pypi
opendatalab               0.0.9                    pypi_0    pypi
openh264                  2.1.1                h780b84a_0    conda-forge
openjpeg                  2.4.0                hb52868f_1    conda-forge
openmim                   0.3.9                    pypi_0    pypi
openssl                   1.1.1v               h7f8727e_0  
ordered-set               4.1.0                    pypi_0    pypi
overrides                 7.3.1                    pypi_0    pypi
packaging                 23.1                     pypi_0    pypi
pandas                    2.0.3                    pypi_0    pypi
pandocfilters             1.5.0                    pypi_0    pypi
parso                     0.8.3                    pypi_0    pypi
pathspec                  0.11.1                   pypi_0    pypi
pexpect                   4.8.0                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    10.0.0                   pypi_0    pypi
pip                       23.1.2           py38h06a4308_0  
pkgutil-resolve-name      1.3.10                   pypi_0    pypi
platformdirs              3.9.1                    pypi_0    pypi
plotly                    5.15.0                   pypi_0    pypi
pluggy                    1.2.0                    pypi_0    pypi
plyfile                   1.0                      pypi_0    pypi
prometheus-client         0.17.1                   pypi_0    pypi
prompt-toolkit            3.0.39                   pypi_0    pypi
protobuf                  4.23.4                   pypi_0    pypi
psutil                    5.9.5                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.2                    pypi_0    pypi
pyasn1                    0.5.0                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pycocotools               2.0.6                    pypi_0    pypi
pycodestyle               2.9.1                    pypi_0    pypi
pycparser                 2.21                     pypi_0    pypi
pycryptodome              3.18.0                   pypi_0    pypi
pyflakes                  2.5.0                    pypi_0    pypi
pygments                  2.15.1                   pypi_0    pypi
pyparsing                 3.0.9                    pypi_0    pypi
pyquaternion              0.9.9                    pypi_0    pypi
pytest                    7.4.0                    pypi_0    pypi
python                    3.8.0                h0371630_2  
python-dateutil           2.8.2                    pypi_0    pypi
python-json-logger        2.0.7                    pypi_0    pypi
python_abi                3.8                      2_cp38    conda-forge
pytorch                   1.10.0          py3.8_cuda11.3_cudnn8.2.0_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2023.3                   pypi_0    pypi
pywavelets                1.4.1                    pypi_0    pypi
pyyaml                    6.0.1                    pypi_0    pypi
pyzmq                     25.1.0                   pypi_0    pypi
qtconsole                 5.4.3                    pypi_0    pypi
qtpy                      2.3.1                    pypi_0    pypi
readline                  7.0                  h7b6447c_5  
referencing               0.30.0                   pypi_0    pypi
requests                  2.31.0                   pypi_0    pypi
requests-oauthlib         1.3.1                    pypi_0    pypi
rfc3339-validator         0.1.4                    pypi_0    pypi
rfc3986-validator         0.1.1                    pypi_0    pypi
rich                      13.4.2                   pypi_0    pypi
rpds-py                   0.9.2                    pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
scikit-image              0.21.0                   pypi_0    pypi
scikit-learn              1.3.0                    pypi_0    pypi
scipy                     1.10.1                   pypi_0    pypi
send2trash                1.8.2                    pypi_0    pypi
setuptools                67.8.0           py38h06a4308_0  
shapely                   1.8.5                    pypi_0    pypi
six                       1.16.0             pyh6c4a22f_0    conda-forge
sniffio                   1.3.0                    pypi_0    pypi
soupsieve                 2.4.1                    pypi_0    pypi
sparsehash                2.0.2                         0    bioconda
sqlite                    3.33.0               h62c20be_0  
stack-data                0.6.2                    pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
tenacity                  8.2.2                    pypi_0    pypi
tensorboard               2.13.0                   pypi_0    pypi
tensorboard-data-server   0.7.1                    pypi_0    pypi
termcolor                 2.3.0                    pypi_0    pypi
terminado                 0.17.1                   pypi_0    pypi
terminaltables            3.1.10                   pypi_0    pypi
threadpoolctl             3.2.0                    pypi_0    pypi
tifffile                  2023.7.10                pypi_0    pypi
tinycss2                  1.2.1                    pypi_0    pypi
tk                        8.6.12               h1ccaba5_0  
tomli                     2.0.1                    pypi_0    pypi
tomlkit                   0.11.8                   pypi_0    pypi
torch-scatter             2.0.9                    pypi_0    pypi
torchsparse               1.4.0                    pypi_0    pypi
torchvision               0.11.0               py38_cu113    pytorch
tornado                   6.3.2                    pypi_0    pypi
tqdm                      4.65.0                   pypi_0    pypi
traitlets                 5.9.0                    pypi_0    pypi
trimesh                   3.22.4                   pypi_0    pypi
typing_extensions         4.7.1              pyha770c72_0    conda-forge
tzdata                    2023.3                   pypi_0    pypi
uri-template              1.3.0                    pypi_0    pypi
urllib3                   1.26.16                  pypi_0    pypi
wcwidth                   0.2.6                    pypi_0    pypi
webcolors                 1.13                     pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.6.1                    pypi_0    pypi
werkzeug                  2.3.6                    pypi_0    pypi
wheel                     0.38.4           py38h06a4308_0  
widgetsnbextension        4.0.8                    pypi_0    pypi
xz                        5.4.2                h5eee18b_0  
y-py                      0.6.0                    pypi_0    pypi
yapf                      0.40.1                   pypi_0    pypi
ypy-websocket             0.8.4                    pypi_0    pypi
zipp                      3.16.2                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.2                ha4553b6_0 
```

