from collections import defaultdict
from itertools import accumulate

import numpy as np
import torch


def collate_batch_cylinder(data):
    grid_ind_stack = [d[0] for d in data]
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    p_fea = [d[2] for d in data]
    p_label = [d[3] for d in data]
    return grid_ind_stack, torch.from_numpy(label2stack), p_fea, p_label,


def collate_batch_voxel(data):
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