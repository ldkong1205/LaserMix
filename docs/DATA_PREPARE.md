<img src="../docs/figs/logo.png" align="right" width="20%">

# Data Preparation

### Overall Structure

```
└── data 
    └── sets
        └── nuscenes
        └── semantickitti
        └── scribblekitti
        └── waymo_open
```

### nuScenes

To prepare the [nuScenes-lidarseg](https://www.nuscenes.org/nuscenes) dataset, download the data, annotations, and other files from https://www.nuscenes.org/download. Unpack the compressed file(s) into `/data/sets/nuscenes` and your folder structure should end up looking like this:

```
└── nuscenes  
    ├── Usual nuscenes folders (i.e. samples, sweep)
    │
    ├── lidarseg
    │   └── v1.0-{mini, test, trainval} <- contains the .bin files; a .bin file 
    │                                      contains the labels of the points in a 
    │                                      point cloud (note that v1.0-test does not 
    │                                      have any .bin files associated with it)
    │
    └── v1.0-{mini, test, trainval}
        ├── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)
        ├── lidarseg.json  <- contains the mapping of each .bin file to the token   
        └── category.json  <- contains the categories of the labels (note that the 
                              category.json from nuScenes v1.0 is overwritten)
```

<hr>

### SemanticKITTI

To prepare the [SemanticKITTI](http://semantic-kitti.org/index) dataset, download the data, annotations, and other files from http://semantic-kitti.org/dataset. Unpack the compressed file(s) into `/data/sets/semantickitti` and re-organize the data structure. Your folder structure should end up looking like this:

```
└── semantickitti  
    └── sequences
        ├── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 21
        ├── labels   <- contains the .label files; a .label file contains the labels of the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 10
        ├── calib
        │    └── 00
        │    └── ···
        │    └── 21
        └── semantic-kitti.yaml
```

#### :memo: Create SemanticKITTI Dataset
- For fully-supervised training and evaluation:
  - We support scripts that generate dataset information for training and validation. Create these `.pkl` info files by running:
    ```Shell
    python ./tools/create_data.py semantickitti --root-path ./data/semantickitti --out-dir ./data/semantickitti --extra-tag semantickitti
    ```
- For semi-supervised training and evaluation:
  - Download the pre-processed `.pkl` files from [here](https://drive.google.com/drive/folders/1PInw2Wvt-vgNzOxlSd2EiDANrTsWV7w1) and put them under the `semantickitti/` folder.
    ```
    └── semantickitti
        ├── sequences
        ├── semantickitti_infos_train.pkl
        ├── semantickitti_infos_val.pkl
        ├── ...
        ├── semantickitti_infos_train.10.pkl
        ├── semantickitti_infos_train.10-unlabeled.pkl
        └── ...
    ```
<hr>

### ScribbleKITTI

To prepare the [ScribbleKITTI](https://arxiv.org/abs/2203.08537) dataset, download the annotations from https://data.vision.ee.ethz.ch/ouenal/scribblekitti.zip. Note that you only need to download these annotation files (~118.2MB); the data is the same as [SemanticKITTI](http://semantic-kitti.org/index). Unpack the compressed file(s) into `/data/sets/scribblekitti` and re-organize the data structure. Your folder structure should end up looking like this:


```
└── scribblekitti 
    └── sequences
        └── scribbles <- contains the .label files; a .label file contains the scribble labels of the points in a point cloud
             └── 00
             └── ···
             └── 10
```

<hr>

### Waymo Open

Coming soon.

<hr>

### References

Please note that you should cite the corresponding paper(s) once you use these datasets.

#### nuScenes
```bibtex
@article{fong2022panopticnuscenes,
    author = {W. K. Fong and R. Mohan and J. V. Hurtado and L. Zhou and H. Caesar and O. Beijbom and A. Valada},
    title = {Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
    journal = {IEEE Robotics and Automation Letters},
    volume = {7},
    number = {2},
    pages = {3795--3802},
    year = {2022}
}
```
```bibtex
@inproceedings{caesar2020nuscenes,
    author = {H. Caesar and V. Bankiti and A. H. Lang and S. Vora and V. E. Liong and Q. Xu and A. Krishnan and Y. Pan and G. Baldan and O. Beijbom},
    title = {nuScenes: A Multimodal Dataset for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {11621--11631},
    year = {2020}
}
```

#### SemanticKITTI

```bibtex
@inproceedings{behley2019semantickitti,
    author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
    title = {SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages = {9297--9307},
    year = {2019}
}
```
```bibtex
@inproceedings{geiger2012kitti,
    author = {A. Geiger and P. Lenz and R. Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {3354--3361},
    year = {2012}
}
```

#### ScribbleKITTI

```bibtex
@inproceedings{unal2022scribble,
    author = {O. Unal and D. Dai and L. Van Gool},
    title = {Scribble-Supervised LiDAR Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {2697--2707},
    year = {2022}
}
```

#### Waymo Open

```bibtex
@inproceedings{sun2020waymoopen,
    author = {P. Sun and H. Kretzschmar and X. Dotiwalla and A. Chouard and V. Patnaik and P. Tsui and J. Guo and Y. Zhou and Y. Chai and B. Caine and V. Vasudevan and W. Han and J. Ngiam and H. Zhao and A. Timofeev and S. Ettinger and M. Krivokon and A. Gao and A. Joshi and Y. Zhang and J. Shlens and Z. Chen and D. Anguelov},
    title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {2446--2454},
    year = {2020}
}
```



