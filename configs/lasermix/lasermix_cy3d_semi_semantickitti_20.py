_base_ = [
    '../_base_/datasets/semi_semantickitti_seg.py',
    '../_base_/schedules/schedule-3x.py', '../_base_/default_runtime.py'
]

dataset_type = 'SemanticKittiDataset'
data_root = 'data/semantickitti/'
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = {
    0: 19,  # "unlabeled"
    1: 19,  # "outlier" mapped to "unlabeled" --------------mapped
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------mapped
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------mapped
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10,  # "sidewalk"
    49: 11,  # "other-ground"
    50: 12,  # "building"
    51: 13,  # "fence"
    52: 19,  # "other-structure" mapped to "unlabeled" ------mapped
    60: 8,  # "lane-marking" to "road" ---------------------mapped
    70: 14,  # "vegetation"
    71: 15,  # "trunk"
    72: 16,  # "terrain"
    80: 17,  # "pole"
    81: 18,  # "traffic-sign"
    99: 19,  # "other-object" to "unlabeled" ----------------mapped
    252: 0,  # "moving-car" to "car" ------------------------mapped
    253: 6,  # "moving-bicyclist" to "bicyclist" ------------mapped
    254: 5,  # "moving-person" to "person" ------------------mapped
    255: 7,  # "moving-motorcyclist" to "motorcyclist" ------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehic------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------mapped
    258: 3,  # "moving-truck" to "truck" --------------------mapped
    259: 4  # "moving-other"-vehicle to "other-vehicle"-----mapped
}

metainfo = dict(classes=class_names, seg_label_mapping=labels_map, max_label=259)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None
branch_field = ['sup', 'unsup']

# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadAnnotations3D',  with_bbox_3d=False, with_label_3d=False, with_seg_3d=True, seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti', backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05], translation_std=[0.1, 0.1, 0.1]),
    dict(type='MultiBranch3D', branch_field=branch_field, sup=dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']))
]

# pipeline used to augment unlabeled data,
# which will be sent to teacher model for predicting pseudo instances.
unsup_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05], translation_std=[0.1, 0.1, 0.1]),
    dict(type='MultiBranch3D', branch_field=branch_field, unsup=dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']))
]

grid_shape = [240, 180, 20]

segmentor = dict(
    type='Cylinder3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor', voxel=True, voxel_type='cylindrical', voxel_layer=dict(grid_shape=grid_shape, point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2], max_num_points=-1, max_voxels=-1)),
    voxel_encoder=dict(type='SegVFE', feat_channels=[64, 128, 256, 256], in_channels=6, with_voxel_center=True, feat_compression=16, grid_shape=grid_shape),
    backbone=dict(type='Asymm3DSpconv', grid_size=grid_shape, input_channels=16, base_channels=32, norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='Cylinder3DHead', channels=128, num_classes=20,
        loss_ce=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, class_weight=None, loss_weight=1.0),  # ce loss weight
        loss_lovasz=dict(type='LovaszLoss', loss_weight=3.0, reduction='none')  # lovasz loss weight
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole'))

model = dict(
    type='LaserMix', segmentor_student=segmentor, segmentor_teacher=segmentor,
    data_preprocessor=dict(
        type='MultiBranch3DDataPreprocessor',
        data_preprocessor=dict(type='Det3DDataPreprocessor', voxel=True, voxel_type='cylindrical', voxel_layer=dict(grid_shape=grid_shape, point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2], max_num_points=-1, max_voxels=-1))
    ),
    loss_mse=(dict(type='mmdet.MSELoss', loss_weight=200)),
    semi_train_cfg=dict(
        freeze_teacher=True, pseudo_thr=0.9, ignore_label=19,
        pitch_angles=[-25, 3], num_areas=[4, 5, 6, 7, 8],
        sup_weight=1, unsup_weight=2,
    ),
    semi_test_cfg=dict(extract_feat_on='teacher', predict_on='teacher'))

# quota
labeled_dataset = dict(
    ann_file='semantickitti_infos_train.20.pkl'
)
unlabeled_dataset = dict(
    ann_file='semantickitti_infos_train.20-unlabeled.pkl',
)
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler', batch_size=4, source_ratio=[1, 1]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset])
)

# learning rate
lr = 0.008  # max learning rate
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01), clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = [  # 45000 iters for 8xb2, 23000 iters for 8xb4
    dict(type='OneCycleLR', total_steps=45000, by_epoch=False, eta_max=0.001)
]

train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop',
    max_iters=45000,  # 45000 iters for 8xb2, 23000 iters for 8xb4
    val_interval=1000)

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=1))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='mmdet.MeanTeacherHook', momentum=0.01)]
