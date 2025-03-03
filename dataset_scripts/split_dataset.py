import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


### Split dataset in one folder into train and val ###


img_folder = "/home/techtrans2/RAT_DATASETS/LAB_RAT_KPTS_DATASET/images"
#label_folder = "/home/cv-worker/Chizh_datasets_people/val/labels"

train_images_folder = "/home/techtrans2/RAT_DATASETS/LAB_RAT_KPTS_DATASET/WISTAR_RAT_KPTS_DATASET_COCO/train/images"
val_images_folder = "/home/techtrans2/RAT_DATASETS/LAB_RAT_KPTS_DATASET/WISTAR_RAT_KPTS_DATASET_COCO/val/images"
#train_labels_folder = "/home/cv-worker/Chizh_datasets_people/train/labels"
#val_labels_folder = "/home/cv-worker/Chizh_datasets_people/test/labels"


images = sorted(os.listdir(img_folder))  # find all images and labels files
#labels = sorted(os.listdir(label_folder))
# split in train and val
#train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.33)
train_images, val_images = train_test_split(images, test_size=0.2)


# for img, label in tqdm(zip(train_images, train_labels)):
#     shutil.copy(img_folder + "/" + img, train_images_folder + "/" + img)  # copy train images
#     shutil.copy(label_folder + "/" + label, train_labels_folder + "/" + label)  # copy train labels
    
# for img, label in tqdm(zip(val_images, val_labels)):
#     shutil.move(img_folder + "/" + img, val_images_folder + "/" + img)  # copy val images
#     shutil.move(label_folder + "/" + label, val_labels_folder + "/" + label)  # copy val labels


for img in tqdm(train_images):
    shutil.copy(img_folder + "/" + img, train_images_folder + "/" + img)  # copy train images
    
for img in tqdm(val_images):
    shutil.copy(img_folder + "/" + img, val_images_folder + "/" + img)  # copy val images
