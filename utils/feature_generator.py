import numpy as np
from typing import Tuple, Dict, List
from .utils import pose_acceleration, pose_speed, pairwise_distances, angle, bbox


class FeatureGenerator:
    def __init__(self, config: Dict) -> None:
        self.ds_config = config
        self.n_kpts = len(config['kpts'])
        self.selected_ids = config['selected_ids']
        self.angle_pairs = config['angle_pairs']
        self.features = config['features']

    def build_feature_matrix(self, kpts: np.ndarray) -> np.ndarray:
        assert len(kpts.shape) == 2, f"kpts must have shape (n_frames, n_kpts * 2) but got shape {kpts.shape} instead"
        assert kpts.shape[1] % 2 == 0, f"n_kpts * 2 must be even but got n_kpts * 2 = {kpts.shape[1]} instead"
        n_frames = kpts.shape[0]
        if 'bbox' in self.features:
            bboxes = kpts[:, :4]
            kpts = kpts[:, 4:]
        kpts = kpts.reshape((n_frames, -1, 2))
        out_features = []
        selected_kpts = kpts[:, self.selected_ids, :].reshape((n_frames, -1))  # select keypoints
        if 'position' in self.features:
            out_features.append(selected_kpts)
        if 'bbox' in self.features:
            out_features.append(bbox(bboxes))
        if 'angles' in self.features and len(self.angle_pairs):
            angles = np.concatenate([angle(kpts[:, pair[1], :] - kpts[:, pair[0], :]) for pair in self.angle_pairs], axis=1)
            out_features.append(angles)
        if 'distance' in self.features:
            out_features.append(pairwise_distances(selected_kpts))
        if 'speed' in self.features:
            out_features.append(pose_speed(selected_kpts))
        if 'acceleration' in self.features:
            out_features.append(pose_acceleration(selected_kpts))
        return np.concatenate(out_features, axis=1)
    
    def build_stat_feature_vector(self, f_matrix: np.ndarray,
                                  n_subwindows: int = 2) -> np.ndarray:
        # Build feature vector for gradient boosting classifier
        # Compute and concatenate features mean, min, max, and std inside provided window and subwindows
        w_size = f_matrix.shape[0]
        assert w_size % 2 == 0, f"size of window must be even but got {w_size} instead"
        #print(f_matrix.shape)
        f_vector = np.concatenate((f_matrix.mean(axis=0), f_matrix.min(axis=0), 
                                f_matrix.max(axis=0), f_matrix.std(axis=0)))
        #print(f_vector.shape)
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
