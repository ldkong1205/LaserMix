import cv2
import glob
import random
import yaml
from collections import defaultdict

import numba as nb
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import functional as F

from data.semantickitti.laserscan import SemLaserScan
from data.semantickitti.pointcloud import SemkittiDataset


class SemkittiRangeViewDatabase(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        data_split: str = None,
        range_img_size: tuple = (64, 1920),
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

        if self.split == 'train':
            folders = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            folders = ['08']
        elif self.split == 'test':
            folders = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        else:
            raise NotImplementedError

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

    def __getitem__(self, index: int):
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

        return {
            'scan': F.to_tensor(scan),
            'label': F.to_tensor(label).to(dtype=torch.long),
            'name': self.lidar_list[index],
        }

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

    def sem_label_transform(self, raw_label_map: np.ndarray):
        for i in self.label_transfer_dict.keys():
            raw_label_map[raw_label_map==i] = self.label_transfer_dict[i]
        
        return raw_label_map

    def generate_label(self, semantic_label: np.ndarray):
        original_label = np.copy(semantic_label)
        label_new = self.sem_label_transform(original_label)
        
        return label_new

    def fill_spherical(self, range_image: np.ndarray):
        height, width = np.shape(range_image)[:2]
        value_mask = np.asarray(1.0-np.squeeze(range_image) > 0.1).astype(np.uint8)
        dt, lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)
        with_value = np.squeeze(range_image) > 0.1
        depth_list = np.squeeze(range_image)[with_value]
        label_list = np.reshape(lbl, [1, height * width])
        depth_list_all = depth_list[label_list - 1]
        depth_map = np.reshape(depth_list_all, (height, width))
        depth_map = cv2.GaussianBlur(depth_map, (7, 7), 0)
        depth_map = range_image * with_value + depth_map * (1 - with_value)
        
        return depth_map


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


class SemkittiCylinderDatabase(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        data_split: str = None,
        voxel_grid_size: list = [240, 180, 32],
        max_volume_space: list = [50, 180, 2],
        min_volume_space: list = [0, -180, -4],
        augment: str = 'NoAugment',
        if_scribble: bool = False,
        if_sup_only: bool = False,
    ):
        super().__init__()
        self.root = root
        self.voxel_grid_size = np.array(voxel_grid_size)
        self.augment = augment

        self.point_cloud_dataset = SemkittiDataset(
            root=self.root,
            split=split,
            data_split=data_split,
            if_scribble=if_scribble,
            if_sup_only=if_sup_only,
        )

        if self.augment == 'GlobalAugment':
            self.if_drop, self.if_flip, self.if_scale, self.if_rotate, self.if_jitter = True, True, True, True, True
        elif self.augment == 'NoAugment':
            self.if_drop, self.if_flip, self.if_scale, self.if_rotate, self.if_jitter = False, False, False, False, False
        else:
            raise NotImplementedError

        self.ignore_label = 0
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        pc_data = self.point_cloud_dataset[index]

        points = pc_data['scan'][:, :3]  # x, y, z
        intensity = pc_data['scan'][:, 3]  # intensity
        ring = pc_data['scan'][:, 4]  # ring
        labels = pc_data['label']

        # data aug (drop)
        if self.if_drop:
            max_num_drop = int(len(points) * 0.1)  # drop ~10%
            num_drop = np.random.randint(low=0, high=max_num_drop)
            self.points_to_drop = np.random.randint(low=0, high=len(points)-1, size=num_drop)
            self.points_to_drop = np.unique(self.points_to_drop)
            points = np.delete(points, self.points_to_drop, axis=0)
            intensity = np.delete(intensity, self.points_to_drop)
            ring = np.delete(ring, self.points_to_drop)
            labels = np.delete(labels, self.points_to_drop)

        # data aug (flip)
        if self.if_flip:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]  # flip x
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]  # flip y
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]  # flip both x and y
        
        # data aug (scale)
        if self.if_scale:
            scale = 1.05  # [-5%, +5%]
            rand_scale = np.random.uniform(1, scale)
            if np.random.random() < 0.5:
                rand_scale = 1 / scale
            points[:, 0] *= rand_scale
            points[:, 1] *= rand_scale

        # data aug (rotate)
        if self.if_rotate:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)

        # data aug (jitter)
        if self.if_jitter:
            jitter = 0.1
            rand_jitter = np.clip(np.random.normal(0, jitter, 3), -3 * jitter, 3 * jitter)
            points += rand_jitter

        xyz_pol = self.cart2polar(points)
        xyz_pol[:, 1] = xyz_pol[:, 1] / np.pi * 180.0

        max_bound = np.asarray(self.max_volume_space)
        min_bound = np.asarray(self.min_volume_space)

        crop_range = max_bound - min_bound
        cur_grid_size = self.voxel_grid_size
        intervals = crop_range / (cur_grid_size - 1)

        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        processed_label = np.ones(self.voxel_grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, np.expand_dims(labels, axis=-1)], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, points[:, :2]), axis=1)
        return_fea = np.concatenate((return_xyz, np.expand_dims(intensity, axis=-1), np.expand_dims(ring, axis=-1)), axis=1)
        
        return {
            'voxel_label': processed_label,
            'grid_ind': grid_ind,
            'point_label': labels,
            'point_feature': return_fea,
        }

    def cart2polar(self, input_xyz):
        rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

    def polar2cat(self, input_xyz_polar):
        x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
        y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
        return np.stack((x, y, input_xyz_polar[2]), axis=0)

    # @staticmethod
    # def collate_batch(batch_list):
    #     data_dict = defaultdict(list)
    #     for cur_sample in batch_list:
    #         for key, val in cur_sample.items():
    #             data_dict[key].append(val)
    #     batch_size = len(batch_list)
    #     ret = {}
    #     ret['voxel_label'] = torch.from_numpy(np.stack(data_dict['voxel_label']).astype(np.int))

    #     grid_ind = []
    #     for i_batch in range(batch_size):
    #         grid_ind.append(
    #             np.pad(data_dict['grid_ind'][i_batch], ((0, 0), (1, 0)), mode='constant', constant_values=i_batch)
    #         )
    #     ret['grid_ind'] = torch.from_numpy(np.concatenate(grid_ind))
    #     ret['point_label'] = torch.from_numpy(np.concatenate(data_dict['point_label']))
    #     ret['point_feature'] = torch.from_numpy(np.concatenate(data_dict['point_feature'])).type(torch.FloatTensor)

    #     return ret
