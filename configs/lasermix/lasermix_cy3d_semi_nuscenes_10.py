_base_ = [
    '../_base_/datasets/semi_nuscenes_seg.py',
    '../_base_/schedules/schedule-3x.py', '../_base_/default_runtime.py'
]

dataset_type = 'NuScenesSegDataset'
data_root = '/data/sets/nuScenes/'

class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]
labels_map = {
    0:  16,
    1:  16,
    2:  6,
    3:  6,
    4:  6,
    5:  16,
    6:  6,
    7:  16,
    8:  16,
    9:  0,
    10: 16,
    11: 16,
    12: 7,
    13: 16,
    14: 1,
    15: 2,
    16: 2,
    17: 3,
    18: 4,
    19: 16,
    20: 16,
    21: 5,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: 16,
    30: 15,
    31: 16
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=31)

input_modality = dict(use_lidar=True, use_camera=False)

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    img='',
    pts_semantic_mask='lidarseg/v1.0-trainval')

backend_args = None

branch_field = ['sup', 'unsup']

randomness = dict(seed=1205, deterministic=False, diff_rank_seed=True)

# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR', load_dim=5, use_dim=4, backend_args=backend_args
    ),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=False, with_label_3d=False, with_seg_3d=True, seg_3d_dtype='np.uint8', backend_args=backend_args
    ),
    dict(type='PointSegClassMapping'),
    dict(type='RandomFlip3D',
         sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5
    ),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05], translation_std=[0.1, 0.1, 0.1]
    ),
    dict(type='MultiBranch3D',
         branch_field=branch_field, sup=dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
    ),
]

# pipeline used to augment unlabeled data,
# which will be sent to teacher model for predicting pseudo instances.
unsup_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR', load_dim=5, use_dim=4, backend_args=backend_args
    ),
    dict(type='RandomFlip3D',
         sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5
    ),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05], translation_std=[0.1, 0.1, 0.1]
    ),
    dict(type='MultiBranch3D',
         branch_field=branch_field, unsup=dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
    ),
]

grid_shape = [240, 180, 20]

segmentor = dict(
    type='Cylinder3D',
    data_preprocessor= dict(type='Det3DDataPreprocessor',
             voxel=True, voxel_type='cylindrical',
             voxel_layer=dict(grid_shape=grid_shape, point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2], max_num_points=-1, max_voxels=-1),
        ),
    voxel_encoder = dict(type='SegVFE',
             feat_channels=[64, 128, 256, 256], in_channels=6, with_voxel_center=True,
             feat_compression=16, grid_shape=grid_shape,
        ),
    backbone = dict(type='Asymm3DSpconv',
             grid_size=grid_shape, input_channels=16, base_channels=32,
             norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1),
        ),
    decode_head = dict(type='Cylinder3DHead',
             channels=128, num_classes=20,
             loss_ce=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, class_weight=None, loss_weight=1),
             loss_lovasz=dict(type='LovaszLoss', loss_weight=2, reduction='none'),
        ),
    train_cfg=None,
    test_cfg=dict(mode='whole'))

model = dict(
    type='LaserMix', segmentor_student=segmentor, segmentor_teacher=segmentor,
    data_preprocessor=dict(
        type='MultiBranch3DDataPreprocessor',
        data_preprocessor=dict(type='Det3DDataPreprocessor', voxel=True, voxel_type='cylindrical', voxel_layer=dict(grid_shape=grid_shape, point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2], max_num_points=-1, max_voxels=-1))
    ),

    loss_mse=(dict(type='mmdet.MSELoss', loss_weight=500)),
    semi_train_cfg=dict(
        freeze_teacher=True, pseudo_thr=0.95, ignore_label=16,
        pitch_angles=[-30, 10], num_areas=[3, 4, 5, 6],
        sup_weight=1, unsup_weight=1,
    ),
    semi_test_cfg=dict(extract_feat_on='teacher', predict_on='teacher'))

# quota
labeled_dataset = dict(
    type='NuScenesSegDataset',
    data_root=data_root, pipeline=sup_pipeline, data_prefix=data_prefix, metainfo=metainfo,
    modality=input_modality, ignore_index=16, backend_args=backend_args,
    ann_file='nuscenes_infos_train.10.pkl'
)
unlabeled_dataset = dict(
    type='NuScenesSegDataset',
    data_root=data_root, pipeline=unsup_pipeline, data_prefix=data_prefix, metainfo=metainfo,
    modality=input_modality, ignore_index=16, backend_args=backend_args,
    # ann_file='nuscenes_infos_train.10-unlabeled.pkl',
    ann_file='nuscenes_infos_train.10-unlabeled.pkl',
)
train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler', batch_size=4, source_ratio=[1, 1],
    ),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset],
    )
)

# learning rate
lr = 0.008
optim_wrapper = dict(type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2),
)

param_scheduler = [
    dict(type='OneCycleLR',
         total_steps=60000,  # 80000 iters for 8xb2 or 4xb4; 60000 iters for 8xb4 or 4xb8
         by_epoch=False, eta_max=0.001, 
    )
]

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop',
    max_iters=60000,  # 80000 iters for 8xb2 or 4xb4; 60000 iters for 8xb4 or 4xb8
    val_interval=1200,
)

# default hook
default_hooks = dict(checkpoint=dict(by_epoch=False, save_best='miou', rule='greater'))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='mmdet.MeanTeacherHook', momentum=0.01)]