from typing import Tuple

import numpy as np


def InstMix(
    scan: np.ndarray, label: np.ndarray, mask: np.ndarray,
    scan_: np.ndarray, label_: np.ndarray, mask_: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scan_new = scan.copy()
    label_new = label.copy()
    mask_new = mask.copy()

    pix_bicycle = label_ == 2  # cls: 2 (bicycle)
    if np.sum(pix_bicycle) > 20:
        scan_new[pix_bicycle]  = scan_[pix_bicycle]
        label_new[pix_bicycle] = label_[pix_bicycle]
        mask_new[pix_bicycle]  = mask_[pix_bicycle]
    
    pix_motorcycle = label_ == 3  # cls: 3 (motorcycle)
    if np.sum(pix_motorcycle) > 20:
        scan_new[pix_motorcycle]  = scan_[pix_motorcycle]
        label_new[pix_motorcycle] = label_[pix_motorcycle]
        mask_new[pix_motorcycle]  = mask_[pix_motorcycle]

    pix_truck = label_ == 4  # cls: 4 (truck)
    if np.sum(pix_truck) > 20:
        scan_new[pix_truck]  = scan_[pix_truck]
        label_new[pix_truck] = label_[pix_truck]
        mask_new[pix_truck]  = mask_[pix_truck]

    pix_other_vehicle = label_ == 5  # cls: 5 (other-vehicle)
    if np.sum(pix_other_vehicle) > 20:
        scan_new[pix_other_vehicle]  = scan_[pix_other_vehicle]
        label_new[pix_other_vehicle] = label_[pix_other_vehicle]
        mask_new[pix_other_vehicle]  = mask_[pix_other_vehicle]

    pix_person = label_ == 6  # cls: 6 (person)
    if np.sum(pix_person) > 20:
        scan_new[pix_person]  = scan_[pix_person]
        label_new[pix_person] = label_[pix_person]
        mask_new[pix_person]  = mask_[pix_person]

    pix_bicyclist = label_ == 7  # cls: 7 (bicyclist)
    if np.sum(pix_bicyclist) > 20:
        scan_new[pix_bicyclist]  = scan_[pix_bicyclist]
        label_new[pix_bicyclist] = label_[pix_bicyclist]
        mask_new[pix_bicyclist]  = mask_[pix_bicyclist]

    pix_motorcyclist = label_ == 8  # cls: 8 (motorcyclist)
    if np.sum(pix_motorcyclist) > 20:
        scan_new[pix_motorcyclist]  = scan_[pix_motorcyclist]
        label_new[pix_motorcyclist] = label_[pix_motorcyclist]
        mask_new[pix_motorcyclist]  = mask_[pix_motorcyclist]

    pix_other_ground = label_ == 12  # cls: 12 (other-ground)
    if np.sum(pix_other_ground) > 20:
        scan_new[pix_other_ground]  = scan_[pix_other_ground]
        label_new[pix_other_ground] = label_[pix_other_ground]
        mask_new[pix_other_ground]  = mask_[pix_other_ground]

    pix_other_trunk = label_ == 16  # cls: 16 (trunk)
    if np.sum(pix_other_trunk) > 20:
        scan_new[pix_other_trunk]  = scan_[pix_other_trunk]
        label_new[pix_other_trunk] = label_[pix_other_trunk]
        mask_new[pix_other_trunk]  = mask_[pix_other_trunk]
    
    pix_pole = label_ == 18  # cls: 18 (pole)
    if np.sum(pix_pole) > 20:
        scan_new[pix_pole]  = scan_[pix_pole]
        label_new[pix_pole] = label_[pix_pole]
        mask_new[pix_pole]  = mask_[pix_pole]

    pix_traffic_sign = label_ == 19  # cls: 19 (traffic-sign)
    if np.sum(pix_traffic_sign) > 20:
        scan_new[pix_traffic_sign]  = scan_[pix_traffic_sign]
        label_new[pix_traffic_sign] = label_[pix_traffic_sign]
        mask_new[pix_traffic_sign]  = mask_[pix_traffic_sign]

    return scan_new, label_new, mask_new
