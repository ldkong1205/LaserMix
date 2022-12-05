from typing import Dict, Tuple
from nuscenes import NuScenes


class NuscLidarSegLabelMappings:

    @staticmethod
    def ignore_label():
        return 0

    @staticmethod
    def raw_mappings(db: NuScenes) -> dict:
        return {
            'official': {
                'global2local': {
                    'noise': 'ignore_label',
                    'human.pedestrian.adult': 'pedestrian',
                    'human.pedestrian.child': 'pedestrian',
                    'human.pedestrian.wheelchair': 'ignore_label',
                    'human.pedestrian.stroller': 'ignore_label',
                    'human.pedestrian.personal_mobility': 'ignore_label',
                    'human.pedestrian.police_officer': 'pedestrian',
                    'human.pedestrian.construction_worker': 'pedestrian',
                    'animal': 'ignore_label',
                    'vehicle.car': 'car',
                    'vehicle.motorcycle': 'motorcycle',
                    'vehicle.bicycle': 'bicycle',
                    'vehicle.bus.bendy': 'bus',
                    'vehicle.bus.rigid': 'bus',
                    'vehicle.truck': 'truck',
                    'vehicle.construction': 'construction_vehicle',
                    'vehicle.emergency.ambulance': 'ignore_label',
                    'vehicle.emergency.police': 'ignore_label',
                    'vehicle.trailer': 'trailer',
                    'movable_object.barrier': 'barrier',
                    'movable_object.trafficcone': 'traffic_cone',
                    'movable_object.pushable_pullable': 'ignore_label',
                    'movable_object.debris': 'ignore_label',
                    'static_object.bicycle_rack': 'ignore_label',
                    'flat.driveable_surface': 'driveable_surface',
                    'flat.sidewalk': 'sidewalk',
                    'flat.terrain': 'terrain',
                    'flat.other': 'flat_other',
                    'static.manmade': 'manmade',
                    'static.vegetation': 'vegetation',
                    'static.other': 'ignore_label',
                    'vehicle.ego': 'ignore_label'
                },
                'local2id': {
                    'ignore_label': NuscLidarSegLabelMappings.ignore_label(),
                    'barrier': 1,
                    'bicycle': 2,
                    'bus': 3,
                    'car': 4,
                    'construction_vehicle': 5,
                    'motorcycle': 6,
                    'pedestrian': 7,
                    'traffic_cone': 8,
                    'trailer': 9,
                    'truck': 10,
                    'driveable_surface': 11,
                    'flat_other': 12,
                    'sidewalk': 13,
                    'terrain': 14,
                    'manmade': 15,
                    'vegetation': 16,
                },
            },
        }
