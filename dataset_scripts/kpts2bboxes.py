from tqdm import tqdm
import os
import numpy as np
import cv2


### Convert kpts labels in YOLO format to bbox only labels ###
kpts_labels = 'labels_kpts'
det_labels = 'labels_detection'


labels = sorted(os.listdir(kpts_labels))


for label in tqdm(labels, total=len(labels)):
    with open(f"{kpts_labels}/{label}", 'r') as f:
        kpts = f.readline().strip('\n').split(' ')
    kpts = np.array(kpts)
    box = kpts[:5]
    cl, x, y, w, h = box
    out = f"{cl} {x} {y} {w} {h}\n"
    with open(f"{det_labels}/{label}", 'w') as f:
        f.write(out)



