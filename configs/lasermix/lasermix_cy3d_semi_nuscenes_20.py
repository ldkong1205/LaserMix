_base_ = [
    './lasermix_cy3d_8xb4_semi_nuscenes_10.py',
]

# quota
labeled_dataset = _base_.labeled_dataset
labeled_dataset.ann_file = 'nuscenes_infos_train.20.pkl'

unlabeled_dataset = _base_.unlabeled_dataset
unlabeled_dataset.ann_file='nuscenes_infos_train.20-unlabeled.pkl'

train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler', batch_size=4, source_ratio=[1, 1],
    ),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset],
    )
)