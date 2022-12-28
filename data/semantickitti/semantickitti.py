import cv2
import glob
import random
import yaml
from collections import defaultdict
from itertools import accumulate

import numba as nb
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import functional as F

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn

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

        else:
            print("Loading '{}' labeled samples from SemanticKITTI under '{}' split ...".format(
                len(set(self.lidar_list)), self.split)
            )

        if self.if_scribble:
            self.label_list = [i.replace("SemanticKITTI", "ScribbleKITTI") for i in self.label_list]
            self.label_list = [i.replace("labels", "scribbles") for i in self.label_list]
            print("Loading '{}' weakly-annotated labels from ScribbleKITTI under '{}' split ...".format(
                len(set(self.label_list)), self.split)
            )

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


class SemkittiCylinderDatabase(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        data_split: str = None,
        max_volume_space: list = [50, 180, 2],
        min_volume_space: list = [0, -180, -4],
        voxel_grid_size: list = [240, 180, 32],
        augment: str = 'NoAugment',
        if_scribble: bool = False,
        if_sup_only: bool = False,
        n_cls: int = 20,
        ignore_label: int = 0,
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
            self.if_drop, self.if_flip, self.if_scale, self.if_rotate, self.if_jitter = False, True, True, True, True
        elif self.augment == 'NoAugment':
            self.if_drop, self.if_flip, self.if_scale, self.if_rotate, self.if_jitter = False, False, False, False, False
        else:
            raise NotImplementedError

        self.n_cls = n_cls
        self.ignore_label = ignore_label
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        pc_data = self.point_cloud_dataset[index]

        points = pc_data['scan'][:, :4]  # x, y, z, intensity
        point_label = pc_data['label']  # label

        # data aug (drop)
        if self.if_drop:
            max_num_drop = int(len(points) * 0.1)  # drop ~10%
            num_drop = np.random.randint(low=0, high=max_num_drop)
            self.points_to_drop = np.random.randint(low=0, high=len(points)-1, size=num_drop)
            self.points_to_drop = np.unique(self.points_to_drop)
            points = np.delete(points, self.points_to_drop, axis=0)
            point_label = np.delete(point_label, self.points_to_drop)

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
            points[:, :3] += rand_jitter

        # voxelization
        xyz_pol = self.cart2polar(points)
        xyz_pol[:, 1] = xyz_pol[:, 1] / np.pi * 180.

        max_bound = np.asarray(self.max_volume_space)  # [50, 180,  2]
        min_bound = np.asarray(self.min_volume_space)  # [0, -180, -4]

        crop_range = max_bound - min_bound
        cur_grid_size = self.voxel_grid_size
        intervals = crop_range / (cur_grid_size - 1)

        point_coord = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_coord, voxel_label, inds, inverse_map = voxelize_with_label(
            n_cls=self.n_cls, point_coord=point_coord, point_label=point_label,
        )
        voxel_center = (voxel_coord.astype(np.float32) + 0.5) * intervals + min_bound
        voxel_fea = np.concatenate([voxel_center, xyz_pol[inds], points[inds][:, :2], points[inds][:, 3:]], axis=1)
        point_voxel_centers = (point_coord.astype(np.float32) + 0.5) * intervals + min_bound

        point_fea = np.concatenate([point_voxel_centers, xyz_pol, points[:, :2], points[:, 3:]], axis=1)
        
        return {
            'point_fea': point_fea.astype(np.float32),
            'voxel_fea': voxel_fea.astype(np.float32),
            'point_coord': point_coord.astype(np.float32),
            'voxel_coord': voxel_coord.astype(np.int),
            'point_label': point_label.astype(np.int),
            'voxel_label': voxel_label.astype(np.int),
            'inverse_map': inverse_map.astype(np.int),
            'num_points': np.array([points.shape[0]]),
        }

    @staticmethod
    def collate_batch(data):
        data_dict = defaultdict(list)
        for sample in data:
            for key, val in sample.items():
                data_dict[key].append(val)
        
        batch_size = len(data)
        batch_data = {}
        point_coord = []
        voxel_coord = []

        for i in range(batch_size):
            point_coord.append(
                np.pad(data_dict['point_coord'][i], ((0, 0), (0, 1)),
                mode='constant', constant_values=i)
            )
            voxel_coord.append(
                np.pad(data_dict['voxel_coord'][i], ((0, 0), (0, 1)),
                mode='constant', constant_values=i)
            )

        batch_data['point_coord'] = torch.from_numpy(np.concatenate(point_coord)).type(torch.LongTensor)
        batch_data['voxel_coord'] = torch.from_numpy(np.concatenate(voxel_coord)).type(torch.LongTensor)

        batch_data['point_fea'] = torch.from_numpy(np.concatenate(data_dict['point_fea'])).type(torch.FloatTensor)
        batch_data['voxel_fea'] = torch.from_numpy(np.concatenate(data_dict['voxel_fea'])).type(torch.FloatTensor)

        batch_data['point_label'] = torch.from_numpy(np.concatenate(data_dict['point_label'])).type(torch.LongTensor)
        batch_data['voxel_label'] = torch.from_numpy(np.concatenate(data_dict['voxel_label'])).type(torch.LongTensor)

        batch_data['inverse_map'] = torch.from_numpy(np.concatenate(data_dict['inverse_map'])).type(torch.LongTensor)
        batch_data['num_points'] = torch.from_numpy(np.concatenate(data_dict['num_points'])).type(torch.LongTensor)

        offset = [sample['voxel_coord'].shape[0] for sample in data] 
        batch_data['offset'] = torch.tensor(list(accumulate(offset))).int()

        return batch_data

    def cart2polar(self, input_xyz):
        rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

    def polar2cat(self, input_xyz_polar):
        x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
        y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
        return np.stack((x, y, input_xyz_polar[2]), axis=0)


def voxelize_with_label(n_cls, point_coord, point_label):
    voxel_coord, inds, inverse_map = sparse_quantize(point_coord, return_index=True, return_inverse=True)
    voxel_label_counter = np.zeros([voxel_coord.shape[0], n_cls])

    for ind in range(len(inverse_map)):
        if point_label[ind] != 67:
            voxel_label_counter[inverse_map[ind]][point_label[ind]] += 1
    
    voxel_label = np.argmax(voxel_label_counter, axis=1)

    return voxel_coord, voxel_label, inds, inverse_map


class SemkittiVoxelDatabase(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        data_split: str = None,
        max_volume_space: list = [50, 180, 2],
        min_volume_space: list = [0, -180, -4],
        voxel_grid_size: float = 0.05,
        augment: str = 'NoAugment',
        if_scribble: bool = False,
        if_sup_only: bool = False,
        n_cls: int = 20,
        ignore_label: int = 0,
    ):
        super().__init__()
        self.root = root
        self.voxel_grid_size = voxel_grid_size
        self.augment = augment

        self.point_cloud_dataset = SemkittiDataset(
            root=self.root,
            split=split,
            data_split=data_split,
            if_scribble=if_scribble,
            if_sup_only=if_sup_only,
        )

        if self.augment == 'GlobalAugment':
            self.if_drop, self.if_flip, self.if_scale, self.if_rotate, self.if_jitter = False, True, True, True, True
        elif self.augment == 'NoAugment':
            self.if_drop, self.if_flip, self.if_scale, self.if_rotate, self.if_jitter = False, False, False, False, False
        else:
            raise NotImplementedError

        self.n_cls = n_cls
        self.ignore_label = ignore_label
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        pc_data = self.point_cloud_dataset[index]

        points = pc_data['scan'][:, :4]  # x, y, z, intensity
        point_label = pc_data['label']  # label

        # data aug (drop)
        if self.if_drop:
            max_num_drop = int(len(points) * 0.1)  # drop ~10%
            num_drop = np.random.randint(low=0, high=max_num_drop)
            self.points_to_drop = np.random.randint(low=0, high=len(points)-1, size=num_drop)
            self.points_to_drop = np.unique(self.points_to_drop)
            points = np.delete(points, self.points_to_drop, axis=0)
            point_label = np.delete(point_label, self.points_to_drop)

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
            points[:, :3] += rand_jitter

        # voxelization
        pc_ = np.round(points[:, :3] / self.voxel_grid_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        fea_ = points

        _, inds, inverse_map = sparse_quantize(
            pc_,
            return_index=True,
            return_inverse=True,
        )

        pc = pc_[inds]  # [N, 3]
        fea = fea_[inds]  # [N, 4]
        label = point_label[inds]  # [N,]
        
        point_fea = SparseTensor(fea, pc)
        label = SparseTensor(label, pc)
        label_mapped = SparseTensor(point_label, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        
        return {
            'point_fea': point_fea,
            'point_label': label,
            'point_label_mapped': label_mapped,
            'inverse_map': inverse_map,
        }

    @staticmethod
    def collate_batch(data):
        offset = [sample['point_fea'].C.shape[0] for sample in data]
        batch_data = sparse_collate_fn(data)
        batch_data.update(dict(
            offset=torch.tensor(list(accumulate(offset))).int()
        ))
        return batch_data


