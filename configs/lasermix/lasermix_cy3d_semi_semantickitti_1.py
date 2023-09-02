_base_ = [
    './lasermix_cy3d_semi_semantickitti_10.py',
]

# quota
labeled_dataset = _base_.labeled_dataset
labeled_dataset.ann_file = 'semantickitti_infos_train.1.pkl'

unlabeled_dataset = _base_.unlabeled_dataset
unlabeled_dataset.ann_file='semantickitti_infos_train.1-unlabeled.pkl'

train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler', batch_size=4, source_ratio=[1, 1],
    ),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset],
    )
)