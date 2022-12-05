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


class SemkittiRangeViewDataset(data.Dataset):
    
    def __init__(
        self,
        dataset_cfg = None,
        class_names: list = None,
        training: bool = True,
        root_path: bool = None,
        logger = None,
    ):
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.root = root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.logger = logger
        self.split = self.dataset_cfg.DATA_SPLIT['train'] if self.training else self.dataset_cfg.DATA_SPLIT['test']
        self.H, self.W = self.dataset_cfg.H, self.dataset_cfg.W  # (H, W)
        yaml_file_path = 'pcdet/datasets_seg/SemanticKITTI/semantickitti.yaml'
        self.CFG = yaml.safe_load(open(yaml_file_path, 'r'))
        self.color_dict = self.CFG["color_map"]
        self.label_transfer_dict = self.CFG["learning_map"]  # label mapping
        self.nclasses = len(self.color_dict)  # 34

        # common aug
        self.if_drop = False if not self.training else self.dataset_cfg.IF_DROP
        self.if_flip = False if not self.training else self.dataset_cfg.IF_FLIP
        self.if_scale = False if not self.training else self.dataset_cfg.IF_SCALE
        self.if_rotate = False if not self.training else self.dataset_cfg.IF_ROTATE
        self.if_jitter = False if not self.training else self.dataset_cfg.IF_JITTER

        # range aug
        self.if_range_mix = False if not self.training else self.dataset_cfg.IF_RANGE_MIX
        self.if_range_shift = False if not self.training else self.dataset_cfg.IF_RANGE_SHIFT
        self.if_range_paste = False if not self.training else self.dataset_cfg.IF_RANGE_PASTE
        self.instance_list = [
            'bicycle', 'motorcycle', 'truck' 'other-vehicle', 
            'person', 'bicyclist', 'motorcyclist', 'other-ground', 
            'trunk', 'pole', 'traffic-sign'
        ]
        self.if_range_union = False if not self.training else self.dataset_cfg.IF_RANGE_UNION

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
            if_range_mix = self.if_range_mix,
            if_range_paste=self.if_range_paste,
            if_range_union=self.if_range_union,
        )

        if self.split == 'train': folders = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val': folders = ['08']
        elif self.split == 'test': folders = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        elif self.split == 'train_test': folders = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        elif self.split == 'train_val_test': folders = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        self.lidar_list = []
        for folder in folders:
            self.lidar_list += glob.glob(self.root + 'sequences/' + folder + '/velodyne/*.bin') 
        print("Loading '{}' samples from SemanticKITTI under '{}' split".format(len(self.lidar_list), self.split))

        self.label_list = [i.replace("velodyne", "labels") for i in self.lidar_list]
        self.label_list = [i.replace("bin", "label") for i in self.label_list]

        if self.split == 'train_test':
            root_psuedo_labels = '/mnt/lustre/konglingdong/data/sets/sequences/'
            folders_test = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            for i in self.label_list:
                if i.split('sequences/')[1][:2] in folders_test:
                    i.replace(self.root + 'sequences/', root_psuedo_labels)

        print("Loading '{}' labels from SemanticKITTI under '{}' split.\n".format(len(self.label_list), self.split))

    def __len__(self):
        return len(self.lidar_list)

    def __getitem__(self, index):
        self.A.open_scan(self.lidar_list[index])
        self.A.open_label(self.label_list[index])

        # prepare attributes
        dataset_dict = {}

        dataset_dict['xyz'] = self.A.proj_xyz
        dataset_dict['intensity'] = self.A.proj_remission
        dataset_dict['range_img'] = self.A.proj_range
        dataset_dict['xyz_mask'] = self.A.proj_mask
        
        semantic_label = self.A.proj_sem_label
        semantic_train_label = self.generate_label(semantic_label)
        dataset_dict['semantic_label'] = semantic_train_label

        # data aug (range shift)
        if np.random.random() >= (1 - self.if_range_shift):
            split_point = random.randint(100, self.W-100)
            dataset_dict = self.sample_transform(dataset_dict, split_point)

        scan, label, mask = self.prepare_input_label_semantic_with_mask(dataset_dict)

        if self.if_range_mix > 0 or self.if_range_paste > 0 or self.if_range_union > 0:

            idx = np.random.randint(0, len(self.lidar_list))

            self.A.open_scan(self.lidar_list[idx])
            self.A.open_label(self.label_list[idx])

            dataset_dict_ = {}
            dataset_dict_['xyz'] = self.A.proj_xyz
            dataset_dict_['intensity'] = self.A.proj_remission
            dataset_dict_['range_img'] = self.A.proj_range
            dataset_dict_['xyz_mask'] = self.A.proj_mask

            semantic_label = self.A.proj_sem_label
            semantic_train_label = self.generate_label(semantic_label)
            dataset_dict_['semantic_label'] = semantic_train_label

            # data aug (range shift)
            if np.random.random() >= (1 - self.if_range_shift):
                split_point_ = random.randint(100, self.W-100)
                dataset_dict_ = self.sample_transform(dataset_dict_, split_point_)

            scan_, label_, mask_ = self.prepare_input_label_semantic_with_mask(dataset_dict_)

            # data aug (range mix)
            if np.random.random() >= (1 - self.if_range_mix):
                scan_mix1, label_mix1, mask_mix1, scan_mix2, label_mix2, mask_mix2, s = self.BeamMix.forward(scan, label, mask, scan_, label_, mask_)

                if np.random.random() >= 0.5:
                    scan, label, mask = scan_mix1, label_mix1, mask_mix1
                else:
                    scan, label, mask = scan_mix2, label_mix2, mask_mix2

            # data aug (range paste)
            if np.random.random() >= (1 - self.if_range_paste):
                scan, label, mask = self.RangePaste(scan, label, mask, scan_, label_, mask_)

            # data aug (range union)
            if np.random.random() >= (1 - self.if_range_union):
                scan, label, mask = self.RangeUnion(scan, label, mask, scan_, label_, mask_)

        data_dict = {
            'scan_rv': F.to_tensor(scan),
            'label_rv': F.to_tensor(label).to(dtype=torch.long),
            'mask_rv': F.to_tensor(mask),
            'scan_name': self.lidar_list[index],
        }

        return data_dict

        # return F.to_tensor(scan), F.to_tensor(label).to(dtype=torch.long), F.to_tensor(mask), self.lidar_list[index]


    def RangeUnion(self, scan, label, mask, scan_, label_, mask_):
        pix_empty = mask == 0

        scan_new = scan.copy()
        label_new = label.copy()
        mask_new = mask.copy()

        scan_new[pix_empty]  = scan_[pix_empty]
        label_new[pix_empty] = label_[pix_empty]
        mask_new[pix_empty]  = mask_[pix_empty]
        return scan_new, label_new, mask_new


    def RangePaste(self, scan, label, mask, scan_, label_, mask_):
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
        # fill in spherical image for calculating normal vector
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
