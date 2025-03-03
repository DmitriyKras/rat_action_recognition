import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


### Convert keypoints and bboxes from JSON COCO format to YOLO TXT format (with kpts mapping if necessary) ###


JSON_PATH = 'part4/part_4.json'
TARGET_FOLDER = 'part4/labels'

map_kpts = {i: i for i in range(13)}
#map_kpts = {0: 0, 1: 2, 2: 1, 3: 5, 4: 4, 5: 3, 6: 6, 7: 9, 8: 8, 9: 7, 10: 12, 11: 11, 12: 10}

with open(JSON_PATH, 'r') as f:
    annotations = json.load(f)


def convert_bbox_to_yolo(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    # The round function determines the number of decimal places in (xmin, ymin, xmax, ymax)
    x = round(x * dw, 4)
    w = round(w * dw, 4)
    y = round(y * dh, 4)
    h = round(h * dh, 4)
    return (x, y, w, h)


images = annotations['images']
labels = annotations['annotations']

for img, label in tqdm(zip(images, labels), total=len(images)):
    w, h = img['width'], img['height']
    name = Path(img['file_name']).with_suffix('.txt').name
    bbox = convert_bbox_to_yolo((w, h), label['bbox'])
    bbox = [str(cr) for cr in bbox]
    kpts_old = label['keypoints']
    kpts_new = kpts_old.copy()
    for i in range(label['num_keypoints']):
        j = map_kpts[i]  # get kpt id for old
        kpts_new[i * 3] = round(kpts_old[j * 3] / w, 4)
        kpts_new[i * 3 + 1] = round(kpts_old[j * 3 + 1] / h, 4)
        kpts_new[i * 3 + 2] = kpts_old[j * 3 + 2]
    kpts_new = [str(pt) for pt in kpts_new]
    kpts_str = ' '.join(kpts_new)
    bbox_str = ' '.join(bbox)
    out = f"0 {bbox_str} {kpts_str}\n"
    with open(f"{TARGET_FOLDER}/{name}", 'w') as f:
        f.write(out)
