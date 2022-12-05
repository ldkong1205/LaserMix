import argparse
import logging
import os
import time
import random

import numpy as np
import torch

from script.cfgs.config import cfg, cfg_from_yaml_file
from script.trainer.trainer import train
from script.trainer.utils import get_n_params


def get_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    # == general configs ==
    parser.add_argument('--cfg-dir', type=str, default='script/cfgs/semantickitti/range.sup_only.yaml',
                        help='path to config file.')
    parser.add_argument('--log-dir', 
                        help='path to save log and ckpts.', default='./logs/')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for ddp training.')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    # == dataset configs ==
    parser.add_argument('--dataset', type=str, choices=['nuscenes', 'semantickitti', 'scribblekitti'],
                        help='name of the dataset we are going to use.')
    # -- for nuScenes --
    parser.add_argument('--root_nusc', type=str, 
                        help='file root path for the nuScenes databas.', default='/nvme/share/data/sets/nuScenes')
    parser.add_argument('--horiz_angular_res', type=float,
                        help='resolution of horizontal angular.', default=0.1875)
    # -- for SemanticKITTI --
    parser.add_argument('--root_semkitti', type=str, 
                        help='file root path for the SemanticKITTI database.', default='/nvme/share/data/sets/SemanticKITTI/')
    parser.add_argument('--yaml', type=str, 
                        help='path for yaml configuration file.', default='data/semantickitti/semantickitti.yaml')
    parser.add_argument('--H', type=int, 
                        help='height for range view projection.', default=64)
    parser.add_argument('--W', type=int, 
                        help='width for range view projection.', default=2048)
    # == training configs ==
    parser.add_argument('--amp', action='store_true',
                        help='whether to conduct automatic mixed precision training.')
    parser.add_argument('--resume_from', type=str, default='',
                        help='file path for the saved checkpoint.')
    
    return parser.parse_args()
    
def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():

    # set args and cfgs
    args = get_args()
    cfg_from_yaml_file(args.cfg_dir, cfg)

    # set random seed
    set_random_seed(cfg.GENERAL.SEED, deterministic=args.deterministic)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # set logger
    args.log_dir = os.path.join(args.log_dir, cfg.EXP_NAME + cfg.SUFFIX, cfg.DATA.DATASET + '_' + cfg.DATA.SPLIT)
    if not(os.path.exists(args.log_dir)):
        os.makedirs(args.log_dir)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.log_dir, f'{timestamp}.log'))
    logger = logging.getLogger()

    logger.info(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))
    logger.info(' '.join(f'{k}={v}\n' for k, v in vars(cfg).items()))

    # set dataset
    if cfg.DATA.DATASET == 'nuscenes':
        from nuscenes.nuscenes import NuScenes as nuScenes
        from data.nuscenes.nusc import NuscLidarSegDatabase
        from data.nuscenes.dataset_rv import LidarSegRangeViewDataset

        raw_db = nuScenes('v1.0-trainval', args.root_nusc, True, 0.1)
        database_train = NuscLidarSegDatabase(
            nusc_db=raw_db, 
            label_mapping_name='official',
            split='train',
            min_distance=0.9,
        )
        database_val = NuscLidarSegDatabase(
            nusc_db=raw_db, 
            label_mapping_name='official',
            split='val',
            min_distance=0.9,
        )
        datasets = [
            LidarSegRangeViewDataset(
                database_train,
                augment='GlobalAugment',
                horiz_angular_res=args.horiz_angular_res,
            ),
            LidarSegRangeViewDataset(
                database_val,
                augment='NoAugment',
                horiz_angular_res=args.horiz_angular_res,
            ),
        ]
    
    elif cfg.DATA.DATASET == 'semantickitti' or cfg.DATA.DATASET == 'scribblekitti':
        from data.semantickitti.semantickitti import SemkittiLidarSegDatabase

        datasets = [
            SemkittiLidarSegDatabase(
                args.root_semkitti, 'train', (args.H, args.W),
                data_split=cfg.DATA.SPLIT,
                if_drop=True,
                if_flip=True,
                if_scale=True,
                if_rotate=True,
                if_jitter=True,
                if_scribble=True if cfg.DATA.DATASET == 'scribblekitti' else False,
                if_sup_only=cfg.DATA.IF_SUP_ONLY,
            ),
            SemkittiLidarSegDatabase(
                args.root_semkitti, 'val', (args.H, args.W),
                data_split='full',
                if_drop=False,
                if_flip=False,
                if_scale=False,
                if_rotate=False,
                if_jitter=False,
                if_scribble=False,
                if_sup_only=cfg.DATA.IF_SUP_ONLY,
            ),
        ]
    
    else:
        raise NotImplementedError

    # set model
    if cfg.MODEL.MODALITY == 'range':
        from model.range.fidnet.network import FIDNet
        model = FIDNet(num_class=16+1 if cfg.DATA.DATASET == 'nuscenes' else 19+1)

    elif cfg.MODEL.MODALITY == 'voxel':
        pass

    else:
        raise NotImplementedError
    
    logger.info(model)
    logger.info("Model parameters: '{:.3f} M".format(get_n_params(model)/1e6))

    # turn to trainer
    train(logger, model, datasets, args, cfg, device)

if __name__ == '__main__':
    main()