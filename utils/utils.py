import numpy as np
from typing import Tuple, List
import os
from tqdm import tqdm
from math import floor


def read_label(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    with open(path, 'r') as f:
        labels = f.readlines()
    labels = [label.strip('\n').split(' ') for label in labels]
    w, h = labels.pop(0)
    labels = np.array(labels).astype(float)  # frame_n x1 y1 score1 x2 y2 score2 ... (n_frames, 1 + n_kpts*3)
    return labels, (int(w), int(h))


def read_feature(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        labels = f.readlines()
    labels = [label.strip('\n').split(' ') for label in labels]
    labels = np.array(labels).astype(float)
    return labels


def labels2kpts(labels: np.ndarray, n_kpts: int = 13, bbox: bool = True) -> np.ndarray:
    label = labels[:, 1:].copy()  # strip frame number
    out = np.empty((label.shape[0], 0))
    if bbox:
        out = np.concatenate((out, label[:, :4]), axis=1)  # get bbox
    label = label[:, 4:]
    for i in range(n_kpts):
        out = np.concatenate((out, label[:, i * 3 : i * 3 + 2]), axis=1)  # strip scores
    return out  # x1 y1 x2 y2 ... (n_frames, n_kps * 2)


def bbox(bbox: np.ndarray) -> np.ndarray:
    wh = bbox[:, 2:4] - bbox[:, :2]
    return np.concatenate((
        wh, (wh[:, 0] * wh[:, 1])[..., None]
    ), axis=1)


def centroid(kpts: np.ndarray, ids: List[int]) -> np.ndarray:
    n_frames, _ = kpts.shape
    new_kpts = kpts.reshape((n_frames, -1, 2))
    return np.sum(new_kpts[:, ids, :], axis=1) / len(ids)


def angle(v: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi * 2).reshape(-1, 1)


def joint_angles(kpts: np.ndarray) -> np.ndarray:
    # compute angles between selected pairs of joints
    # joint_ids = [
    #     [3, 0],  # head - nose - head oritntation
    #     [10, 7],  # tail_base - center
    #     [7, 6],  # center - center_shoulders
    #     [6, 3],  # center_shoulders - head
    #     [6, 8],  # center_shoulders - right_paw
    #     [6, 9],  # center_shoulders - left_paw
    #     [10, 11],  # tail_base - right_hip
    #     [10, 12],  # tail_base - left_hip
    # ]
    joint_ids = [
        [3, 0],  # neck - nose - head oritntation
        [4, 2],  # center - neck
        [7, 4],  # tail_base - center
        [7, 5],  # tail_base - right_hip
        [7, 6],  # tail_base - left_hip
    ]
    n_frames, _ = kpts.shape
    new_kpts = kpts.reshape((n_frames, -1, 2))
    angles = np.concatenate([
        angle(new_kpts[:, pair[1], :] - new_kpts[:, pair[0], :]) for pair in joint_ids             
    ], axis=1)
    return angles


def pairwise_distances(kpts: np.ndarray) -> np.ndarray:
    # pairwise distances between all kpts
    n_frames, _ = kpts.shape
    new_kpts = kpts.reshape((n_frames, -1, 2))
    n_kpts = new_kpts.shape[1]
    d_matrix = np.linalg.norm(new_kpts[:, :, None, :] - new_kpts[:, None, :, :], axis=-1)
    inds = np.triu_indices(n_kpts, k=1)
    return np.array([m[inds] for m in d_matrix])


def pose_speed(kpts: np.ndarray) -> np.ndarray:
    # compute speed vx, vy for each point
    n_frames, n_kpts = kpts.shape
    diff = np.diff(kpts, axis=0)
    return np.concatenate((np.zeros((1, n_kpts)), diff), axis=0)


def pose_acceleration(kpts: np.ndarray) -> np.ndarray:
    # compute speed ax, ay for each point
    n_frames, n_kpts = kpts.shape
    diff = np.diff(kpts, n=2, axis=0)
    return np.concatenate((np.zeros((2, n_kpts)), diff), axis=0)


def cage_orientation(pose_c: np.ndarray, cage_corners: np.ndarray,
                     water_c: np.ndarray, food_c: np.ndarray) -> np.ndarray:
    # compute distances between pose centroid and cage objects
    assert cage_corners.shape == (4, 2), "4 corners of the cage must be provided"
    cage_d = np.concatenate([
        np.linalg.norm(pose_c - corner, axis=-1).reshape((-1, 1)) for corner in cage_corners
    ], axis=1)
    return np.concatenate((
        cage_d,
        np.linalg.norm(pose_c - water_c, axis=-1).reshape((-1, 1)),
        np.linalg.norm(pose_c - food_c, axis=-1).reshape((-1, 1))
    ), axis=1)


def feature_matrix(kpts: np.ndarray) -> np.ndarray:
    # compute feature vector
    assert len(kpts.shape) == 2, f"kpts must have shape (n_frames, n_kpts * 2) but got shape {kpts.shape} instead"
    assert kpts.shape[1] % 2 == 0, f"n_kpts * 2 must be even but got n_kpts * 2 = {kpts.shape[1]} instead"
    # head_c = centroid(kpts, [0, 1, 2])
    # hips_c = centroid(kpts, [5, 6, 7])
    new_kpts = kpts.reshape((kpts.shape[0], -1, 2))
    new_kpts = new_kpts[:, [0, 3, 4, 7], :].reshape((kpts.shape[0], -1))
    # return np.concatenate((
    #     kpts,  # x, y position data
    #     angle(head_c - hips_c),  # orientation data
    #     pairwise_distances(kpts),  # distances data
    #     joint_angles(kpts),  # angles data
    #     pose_speed(kpts),  # velocity data
    #     #pose_acceleration(kpts),  # acceleration data
    # ), axis=1)
    return np.concatenate((
        new_kpts,  # x, y position data
        angle(new_kpts[:, :2] - new_kpts[:, 2:4]),  # orientation data
        angle(new_kpts[:, 2:4] - new_kpts[:, 4:6]),  # orientation data
        angle(new_kpts[:, 4:6] - new_kpts[:, 6:8]),  # orientation data
        pairwise_distances(new_kpts),  # distances data
        pose_speed(new_kpts),  # velocity data
        #pose_acceleration(kpts),  # acceleration data
    ), axis=1)


def build_stat_feature_vector(kpts_window: np.ndarray,
                                  n_subwindows: int = 2) -> np.ndarray:
    # Build feature vector for gradient boosting classifier
    # Compute and concatenate features mean, min, max, and std inside provided window and subwindows
    w_size = kpts_window.shape[0]
    assert w_size % 2 == 0, f"size of window must be even but got {w_size} instead"
    f_matrix = feature_matrix(kpts_window)  # compute features in whole window
    f_vector = np.concatenate((f_matrix.mean(axis=0), f_matrix.min(axis=0), 
                               f_matrix.max(axis=0), f_matrix.std(axis=0)))
    if n_subwindows > 0:
        w_c = w_size // 2  # center of the window
        new_w_size = w_size // 2  # new size of the window
        for _ in range(n_subwindows):
            new_f_matrix = f_matrix[w_c - new_w_size // 2 : w_c + new_w_size // 2]  # get data in a subwindow
            new_f_vector = np.concatenate((new_f_matrix.mean(axis=0), new_f_matrix.min(axis=0), 
                                           new_f_matrix.max(axis=0), new_f_matrix.std(axis=0)))
            f_vector = np.concatenate((f_vector, new_f_vector))  # append it to feature vector
            new_w_size = new_w_size // 2
    return f_vector


def generate_stat_dataset(root_path: str, class_names: List[str],
                          w_size: int, n_subwindows: int, save_path: str, overlap: float = 0.3) -> None:
    data = []
    labels_path = 'labels_topviewrodents'
    for id, cl in tqdm(enumerate(class_names), total=len(class_names)):
        label_list = os.listdir(f"{root_path}/{cl}/{labels_path}/")  # get txt paths of saved labels
        for path in tqdm(label_list, total=len(label_list)):
        #for path in label_list:
            label = read_label(f"{root_path}/{cl}/{labels_path}/{path}")  # read label as np.ndarray
            label = labels2kpts(label, n_kpts=8)  # strip frame number and kpts scores
            n_frames = label.shape[0]  # total number of frames
            step = floor((1 - overlap) * w_size)  # step of the sliding window
            n_steps = (n_frames - w_size) // step  + 1  # number of steps for label
            # print('\n', n_frames)
            # print(step)
            # print(n_steps)
            for i in range(n_steps):
                f_vector = build_stat_feature_vector(label[i * step : i * step + w_size], n_subwindows).tolist()
                f_vector.append(id)
                data.append(f_vector)

    data = np.array(data)
    #print(data[:, -1])
    np.save(save_path, data)



