import os
import os.path as osp
from pathlib import Path

import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def get_available_scenes(nusc):
    available_scenes = []
    print(f'total scene num: {len(nusc.scene)}')
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print(f'exist scene num: {len(available_scenes)}')
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False):
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmengine.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)

        mmengine.check_file_exist(lidar_path)

        info = dict()
        info['token'] = sample['token']
        info['lidar_points'] = dict(num_pts_feats=5,
                                    lidar_path=Path(lidar_path).name)

        # info = {
        #     'lidar_path': lidar_path,
        #     'num_features': 5,
        #     'token': sample['token']}

        if not test:
            if 'lidarseg' in nusc.table_names:
                info['pts_semantic_mask_path'] = osp.join(
                    nusc.dataroot,
                    nusc.get('lidarseg', lidar_token)['filename'])
                info['pts_semantic_mask_path'] = Path(
                    info['pts_semantic_mask_path']).name

            if 'panoptic' in nusc.table_names:
                info['pts_panoptic_mask_path'] = osp.join(
                    nusc.dataroot,
                    nusc.get('panoptic', lidar_token)['filename'])
                info['pts_panoptic_mask_path'] = Path(
                    info['pts_panoptic_mask_path']).name

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


root_path = '/mnt/lustre/konglingdong/data/sets/nuScenes'
info_prefix = 'nuscenes'
version = 'v1.0-trainval'

nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

if version == 'v1.0-trainval':
    train_scenes = splits.train
    val_scenes = splits.val
elif version == 'v1.0-test':
    train_scenes = splits.test
    val_scenes = []
elif version == 'v1.0-mini':
    train_scenes = splits.mini_train
    val_scenes = splits.mini_val
else:
    raise ValueError('unknown')

# filter existing scenes.
available_scenes = get_available_scenes(nusc)
available_scene_names = [s['name'] for s in available_scenes]
train_scenes = list(
    filter(lambda x: x in available_scene_names, train_scenes)
)
val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
train_scenes = set([
    available_scenes[available_scene_names.index(s)]['token']
    for s in train_scenes
])
val_scenes = set([
    available_scenes[available_scene_names.index(s)]['token']
    for s in val_scenes
])

test = 'test' in version
train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
    nusc, train_scenes, val_scenes, test
)

METAINFO = {
    'classes':
    ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
     'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
}

metainfo = dict()
metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
metainfo['dataset'] = 'nuscenes'
metainfo['version'] = version
metainfo['info_version'] = '1.1'

train_info = dict(metainfo=metainfo, data_list=train_nusc_infos)
val_info = dict(metainfo=metainfo, data_list=val_nusc_infos)

mmengine.dump(train_info, f'{root_path}/{info_prefix}_infos_train.pkl', 'pkl')
mmengine.dump(val_info, f'{root_path}/{info_prefix}_infos_val.pkl', 'pkl')
