import os
import cv2
import glob
import random
import yaml

import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import functional as F

from data.semantickitti.laserscan import SemLaserScan


class SemkittiLidarSegDatabase(data.Dataset):
    
    def __init__(
        self,
        root: str,
        split: str,
        range_img_size: tuple = (64, 1920),
        data_split: str = None,
        augment: str = 'NoAugment',
        if_scribble: bool = False,
        if_sup_only: bool = False,
    ):
        self.root = root
        self.split = split
        self.H, self.W = range_img_size  # (H, W)
        yaml_file_path = 'data/semantickitti/semantickitti.yaml'
        self.CFG = yaml.safe_load(open(yaml_file_path, 'r'))
        self.color_dict = self.CFG["color_map"]
        self.label_transfer_dict = self.CFG["learning_map"]  # label mapping
        self.nclasses = len(self.color_dict)  # 34
        
        self.augment = augment
        self.if_scribble = if_scribble
        self.if_sup_only = if_sup_only

        if self.augment == 'GlobalAugment':
            self.if_drop, self.if_flip, self.if_scale, self.if_rotate, self.if_jitter = True, True, True, True, True
        elif self.augment == 'NoAugment':
            self.if_drop, self.if_flip, self.if_scale, self.if_rotate, self.if_jitter = False, False, False, False, False
        else:
            raise NotImplementedError

        self.A=SemLaserScan(
            nclasses = self.nclasses,
            sem_color_dict = self.color_dict,
            project = True,
            H = self.H,
            W = self.W, 
            fov_up = 3.0,
            fov_down = -25.0,
            if_drop = self.if_drop,
            if_flip = self.if_flip,
            if_scale = self.if_scale,
            if_rotate = self.if_rotate,
            if_jitter = self.if_jitter,
        )

        if data_split == 'full':
            self.data_split_list_path = None
        elif data_split == '1pct':
            self.data_split_list_path = 'script/split/semantickitti/semantickitti_1pct.txt'
        elif data_split == '10pct':
            self.data_split_list_path = 'script/split/semantickitti/semantickitti_10pct.txt'
        elif data_split == '20pct':
            self.data_split_list_path = 'script/split/semantickitti/semantickitti_20pct.txt'
        elif data_split == '50pct':
            self.data_split_list_path = 'script/split/semantickitti/semantickitti_50pct.txt'
        else:
            raise NotImplementedError

        if self.data_split_list_path:
            with open(self.data_split_list_path, "r") as f:
                data_split_list = f.read().splitlines()
            data_split_list = [i.split('train/')[-1] for i in data_split_list]

        if self.split == 'train': folders = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val': folders = ['08']
        elif self.split == 'test': folders = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        self.lidar_list = []
        for folder in folders:
            self.lidar_list += glob.glob(self.root + 'sequences/' + folder + '/velodyne/*.bin')
        self.label_list = [i.replace("velodyne", "labels") for i in self.lidar_list]
        self.label_list = [i.replace("bin", "label") for i in self.label_list]

        if self.data_split_list_path:
            self.lidar_list_labeled = [self.root + 'sequences/' + i for i in data_split_list]
            self.label_list_labeled = [i.replace("velodyne", "labels") for i in self.lidar_list_labeled]
            self.label_list_labeled = [i.replace("bin", "label") for i in self.label_list_labeled]
            print("Loading '{}' labeled samples ('{:.0f}%') from SemanticKITTI under '{}' split ...".format(
                len(self.lidar_list_labeled), (len(self.lidar_list_labeled) / len(self.label_list)) * 100, self.split)
            )

            self.lidar_list_unlabeled = [i for i in self.lidar_list if i not in self.lidar_list_labeled]
            self.label_list_unlabeled = [i.replace("velodyne", "labels") for i in self.lidar_list_unlabeled]
            self.label_list_unlabeled = [i.replace("bin", "label") for i in self.label_list_unlabeled]
            print("Loading '{}' unlabeled samples ('{:.0f}%') from SemanticKITTI under '{}' split ...".format(
                len(self.lidar_list_unlabeled), (len(self.lidar_list_unlabeled) / len(self.label_list)) * 100, self.split)
            )

            self.lidar_list_labeled = self.lidar_list_labeled * int(np.ceil(len(self.lidar_list_unlabeled) / len(self.lidar_list_labeled)))
            self.label_list_labeled = [i.replace("velodyne", "labels") for i in self.lidar_list_labeled]
            self.label_list_labeled = [i.replace("bin", "label") for i in self.label_list_labeled]

            assert len(self.lidar_list_labeled) == len(self.label_list_labeled)
            assert len(self.lidar_list_unlabeled) == len(self.label_list_unlabeled)

            if self.if_sup_only:
                self.lidar_list = self.lidar_list_labeled
                self.label_list = self.label_list_labeled
            else:
                self.lidar_list = self.lidar_list_unlabeled
                self.label_list = self.label_list_unlabeled

        if self.if_scribble:
            self.label_list = [i.replace("SemanticKITTI", "ScribbleKITTI") for i in self.label_list]
            self.label_list = [i.replace("labels", "scribbles") for i in self.label_list]


    def __len__(self):
        return len(self.lidar_list)

    def __getitem__(self, index):
        self.A.open_scan(self.lidar_list[index])
        self.A.open_label(self.label_list[index])

        data_dict = {}

        data_dict['xyz'] = self.A.proj_xyz
        data_dict['intensity'] = self.A.proj_remission
        data_dict['range_img'] = self.A.proj_range
        data_dict['xyz_mask'] = self.A.proj_mask
        
        semantic_label = self.A.proj_sem_label
        semantic_train_label = self.generate_label(semantic_label)
        data_dict['semantic_label'] = semantic_train_label

        split_point = random.randint(100, self.W-100)
        data_dict = self.sample_transform(data_dict, split_point)

        scan, label, mask = self.prepare_input_label_semantic_with_mask(data_dict)

        return F.to_tensor(scan), F.to_tensor(label).to(dtype=torch.long), F.to_tensor(mask), self.lidar_list[index]


    def InstMix(self, scan, label, mask, scan_, label_, mask_):
        scan_new = scan.copy()
        label_new = label.copy()
        mask_new = mask.copy()

        pix_bicycle = label_ == 2  # cls: 2 (bicycle)
        if np.sum(pix_bicycle) > 20:
            scan_new[pix_bicycle]  = scan_[pix_bicycle]
            label_new[pix_bicycle] = label_[pix_bicycle]
            mask_new[pix_bicycle]  = mask_[pix_bicycle]
        
        pix_motorcycle = label_ == 3  # cls: 3 (motorcycle)
        if np.sum(pix_motorcycle) > 20:
            scan_new[pix_motorcycle]  = scan_[pix_motorcycle]
            label_new[pix_motorcycle] = label_[pix_motorcycle]
            mask_new[pix_motorcycle]  = mask_[pix_motorcycle]

        pix_truck = label_ == 4  # cls: 4 (truck)
        if np.sum(pix_truck) > 20:
            scan_new[pix_truck]  = scan_[pix_truck]
            label_new[pix_truck] = label_[pix_truck]
            mask_new[pix_truck]  = mask_[pix_truck]

        pix_other_vehicle = label_ == 5  # cls: 5 (other-vehicle)
        if np.sum(pix_other_vehicle) > 20:
            scan_new[pix_other_vehicle]  = scan_[pix_other_vehicle]
            label_new[pix_other_vehicle] = label_[pix_other_vehicle]
            mask_new[pix_other_vehicle]  = mask_[pix_other_vehicle]

        pix_person = label_ == 6  # cls: 6 (person)
        if np.sum(pix_person) > 20:
            scan_new[pix_person]  = scan_[pix_person]
            label_new[pix_person] = label_[pix_person]
            mask_new[pix_person]  = mask_[pix_person]

        pix_bicyclist = label_ == 7  # cls: 7 (bicyclist)
        if np.sum(pix_bicyclist) > 20:
            scan_new[pix_bicyclist]  = scan_[pix_bicyclist]
            label_new[pix_bicyclist] = label_[pix_bicyclist]
            mask_new[pix_bicyclist]  = mask_[pix_bicyclist]

        pix_motorcyclist = label_ == 8  # cls: 8 (motorcyclist)
        if np.sum(pix_motorcyclist) > 20:
            scan_new[pix_motorcyclist]  = scan_[pix_motorcyclist]
            label_new[pix_motorcyclist] = label_[pix_motorcyclist]
            mask_new[pix_motorcyclist]  = mask_[pix_motorcyclist]

        pix_other_ground = label_ == 12  # cls: 12 (other-ground)
        if np.sum(pix_other_ground) > 20:
            scan_new[pix_other_ground]  = scan_[pix_other_ground]
            label_new[pix_other_ground] = label_[pix_other_ground]
            mask_new[pix_other_ground]  = mask_[pix_other_ground]

        pix_other_trunk = label_ == 16  # cls: 16 (trunk)
        if np.sum(pix_other_trunk) > 20:
            scan_new[pix_other_trunk]  = scan_[pix_other_trunk]
            label_new[pix_other_trunk] = label_[pix_other_trunk]
            mask_new[pix_other_trunk]  = mask_[pix_other_trunk]
        
        pix_pole = label_ == 18  # cls: 18 (pole)
        if np.sum(pix_pole) > 20:
            scan_new[pix_pole]  = scan_[pix_pole]
            label_new[pix_pole] = label_[pix_pole]
            mask_new[pix_pole]  = mask_[pix_pole]

        pix_traffic_sign = label_ == 19  # cls: 19 (traffic-sign)
        if np.sum(pix_traffic_sign) > 20:
            scan_new[pix_traffic_sign]  = scan_[pix_traffic_sign]
            label_new[pix_traffic_sign] = label_[pix_traffic_sign]
            mask_new[pix_traffic_sign]  = mask_[pix_traffic_sign]

        return scan_new, label_new, mask_new


    def prepare_input_label_semantic_with_mask(self, sample):
        scale_x = np.expand_dims(np.ones([self.H, self.W]) * 50.0, axis=-1).astype(np.float32)
        scale_y = np.expand_dims(np.ones([self.H, self.W]) * 50.0, axis=-1).astype(np.float32)
        scale_z = np.expand_dims(np.ones([self.H, self.W]) * 3.0,  axis=-1).astype(np.float32)
        scale_matrx = np.concatenate([scale_x, scale_y, scale_z], axis=2)

        each_input = [
            sample['xyz'] / scale_matrx,
            np.expand_dims(sample['intensity'], axis=-1), 
            np.expand_dims(sample['range_img']/80.0, axis=-1),
            np.expand_dims(sample['xyz_mask'], axis=-1)
        ]
        input_tensor = np.concatenate(each_input, axis=-1)

        semantic_label = sample['semantic_label'][:, :]
        semantic_label_mask = sample['xyz_mask'][:, :]

        return input_tensor, semantic_label, semantic_label_mask


    def sample_transform(self, dataset_dict, split_point):
        dataset_dict['xyz'] = np.concatenate(
            [dataset_dict['xyz'][:, split_point:, :], dataset_dict['xyz'][:, :split_point, :]], axis=1
        )
        dataset_dict['xyz_mask'] = np.concatenate(
            [dataset_dict['xyz_mask'][:, split_point:], dataset_dict['xyz_mask'][:, :split_point]], axis=1
        )
        dataset_dict['intensity'] = np.concatenate(
            [dataset_dict['intensity'][:, split_point:], dataset_dict['intensity'][:, :split_point]], axis=1
        )
        dataset_dict['range_img'] = np.concatenate(
            [dataset_dict['range_img'][:, split_point:], dataset_dict['range_img'][:, :split_point]], axis=1
        )
        dataset_dict['semantic_label'] = np.concatenate(
            [dataset_dict['semantic_label'][:, split_point:], dataset_dict['semantic_label'][:, :split_point]], axis=1
        )
        return dataset_dict


    def sem_label_transform(self,raw_label_map):
        for i in self.label_transfer_dict.keys():
            raw_label_map[raw_label_map==i]=self.label_transfer_dict[i]
        
        return raw_label_map


    def generate_label(self,semantic_label):
        original_label=np.copy(semantic_label)
        label_new=self.sem_label_transform(original_label)
        
        return label_new


    def fill_spherical(self,range_image):
        height,width=np.shape(range_image)[:2]
        value_mask=np.asarray(1.0-np.squeeze(range_image)>0.1).astype(np.uint8)
        dt, lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)
        with_value=np.squeeze(range_image)>0.1
        depth_list=np.squeeze(range_image)[with_value]
        label_list=np.reshape(lbl,[1,height*width])
        depth_list_all=depth_list[label_list-1]
        depth_map=np.reshape(depth_list_all,(height,width))
        depth_map = cv2.GaussianBlur(depth_map,(7,7),0)
        depth_map=range_image*with_value+depth_map*(1-with_value)
        
        return depth_map
                   

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
