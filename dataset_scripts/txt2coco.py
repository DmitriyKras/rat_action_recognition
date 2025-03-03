import json
from pathlib import Path
import os
#from natsort import natsorted
import numpy as np
from tqdm import tqdm
import cv2


### Convert YOLO TXT labels into COCO JSON format ###


TARGET_JSON_PATH = '/home/techtrans2/RAT_DATASETS/LAB_RAT_KPTS_DATASET/WISTAR_RAT_KPTS_DATASET_COCO/annotations/val.json'
LABELS_FOLDER = '/home/techtrans2/RAT_DATASETS/LAB_RAT_KPTS_DATASET/labels'
IMAGES_FOLDER = '/home/techtrans2/RAT_DATASETS/LAB_RAT_KPTS_DATASET/WISTAR_RAT_KPTS_DATASET_COCO/val/images'
N_KPTS = 13


img_id = 1


images = os.listdir(IMAGES_FOLDER)

images_value = []
annotations_value = []
categories_value= [{'id': 1, 'name': 'rat'}, ]


for img in tqdm(sorted(images)):
    # Process image key
    H, W = cv2.imread(f"{IMAGES_FOLDER}/{img}").shape[:2]
    images_value.append({
        'file_name': img,
        'height': H,
        'width': W,
        'id': img_id
    })
    # Process annotation key
    l_path = Path(img).with_suffix('.txt')
    with open(Path(LABELS_FOLDER) / l_path, 'r') as f:
        kpts = f.readline().strip('\n').split(' ')
    kpts = np.array(kpts).astype(float)
    bbox = kpts[1:5] * np.array((W, H, W, H))
    bbox[:2] -= bbox[2:4] / 2
    bbox = bbox.round(2).tolist()  # get bbox
    kpts = kpts[5:] * np.array((W, H, 1) * N_KPTS)
    kpts = kpts.astype(int).tolist()
    annotations_value.append({
        'segmentation': [],
        'keypoints': kpts,
        'num_keypoints': N_KPTS,
        'area': round(bbox[2] * bbox[3], 2),
        'iscrowd': 0,
        'image_id': img_id,
        'bbox': bbox,
        'category_id': 1,
        'id': img_id
    })
    img_id += 1
    

with open(TARGET_JSON_PATH, 'w') as f:
    json.dump({
        'images': images_value,
        'annotations': annotations_value,
        'categories': categories_value
    }, f)
