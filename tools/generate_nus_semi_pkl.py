import mmengine

pkl_file = '/data/sets/nuScenes/nuscenes_infos_train.pkl'

info = mmengine.load(pkl_file)
metainfo = info['metainfo']
data_list = info['data_list']

labeled_1_data_list = []
unlabeled_1_data_list = []

labeled_10_data_list = []
unlabeled_10_data_list = []

labeled_20_data_list = []
unlabeled_20_data_list = []

labeled_50_data_list = []
unlabeled_50_data_list = []

with open('/data/sets/nuScenes/nuscenes_1pct.txt', 'r') as f:
    tokens_1 = f.read().splitlines()

with open('/data/sets/nuScenes/nuscenes_10pct.txt', 'r') as f:
    tokens_10 = f.read().splitlines()

with open('/data/sets/nuScenes/nuscenes_20pct.txt', 'r') as f:
    tokens_20 = f.read().splitlines()

with open('/data/sets/nuScenes/nuscenes_50pct.txt', 'r') as f:
    tokens_50 = f.read().splitlines()

for sub_data_list in mmengine.track_iter_progress(data_list):
    if sub_data_list['token'] in tokens_1:
        labeled_1_data_list.append(sub_data_list)
    else:
        unlabeled_1_data_list.append(sub_data_list)

    if sub_data_list['token'] in tokens_10:
        labeled_10_data_list.append(sub_data_list)
    else:
        unlabeled_10_data_list.append(sub_data_list)

    if sub_data_list['token'] in tokens_20:
        labeled_20_data_list.append(sub_data_list)
    else:
        unlabeled_20_data_list.append(sub_data_list)

    if sub_data_list['token'] in tokens_50:
        labeled_50_data_list.append(sub_data_list)
    else:
        unlabeled_50_data_list.append(sub_data_list)

labeled_1_info = dict(metainfo=metainfo, data_list=labeled_1_data_list)
unlabeled_1_info = dict(metainfo=metainfo, data_list=unlabeled_1_data_list)

labeled_10_info = dict(metainfo=metainfo, data_list=labeled_10_data_list)
unlabeled_10_info = dict(metainfo=metainfo, data_list=unlabeled_10_data_list)

labeled_20_info = dict(metainfo=metainfo, data_list=labeled_20_data_list)
unlabeled_20_info = dict(metainfo=metainfo, data_list=unlabeled_20_data_list)

labeled_50_info = dict(metainfo=metainfo, data_list=labeled_50_data_list)
unlabeled_50_info = dict(metainfo=metainfo, data_list=unlabeled_50_data_list)

mmengine.dump(labeled_1_info, 'nuscenes_infos_train.1.pkl', 'pkl')
mmengine.dump(unlabeled_1_info, 'nuscenes_infos_train.1-unlabeled.pkl', 'pkl')

mmengine.dump(labeled_10_info, 'nuscenes_infos_train.10.pkl', 'pkl')
mmengine.dump(unlabeled_10_info, 'nuscenes_infos_train.10-unlabeled.pkl', 'pkl')

mmengine.dump(labeled_20_info, 'nuscenes_infos_train.20.pkl', 'pkl')
mmengine.dump(unlabeled_20_info, 'nuscenes_infos_train.20-unlabeled.pkl', 'pkl')

mmengine.dump(labeled_50_info, 'nuscenes_infos_train.50.pkl', 'pkl')
mmengine.dump(unlabeled_50_info, 'nuscenes_infos_train.50-unlabeled.pkl', 'pkl')
