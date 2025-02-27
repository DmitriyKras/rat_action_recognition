import cv2
import numpy as np
from skimage.transform import resize
from typing import Tuple, Dict, List, Optional, Union
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


class FarnebackFeatureGenerator:
    def __init__(self, of_params: Dict, flow_shape: Tuple[int, int], 
                 frame: np.ndarray, w_size: int) -> None:
        self.of_params = of_params  # optical flow params dict
        self.flow_shape = flow_shape  # size of output fmap
        self.prev_frame = cv2.cvtColor(cv2.resize(frame, self.flow_shape), cv2.COLOR_BGR2GRAY)  # initial frame
        self.out = []
        self.w_size = w_size

    def step(self, frame: np.ndarray, roi_shape: Optional[Tuple[int, int]] = None,
                          bbox: Optional[np.ndarray] = None) -> Union[np.ndarray, None]:
        if bbox is not None:
            assert bbox.shape[0] == 4
            assert roi_shape is not None

        cur = cv2.cvtColor(cv2.resize(frame, self.flow_shape), cv2.COLOR_BGR2GRAY)  # start with first frame
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame, cur, None, **self.of_params)  # calculate dense optical flow
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            flow = flow[y1 : y2, x1 : x2, :]
            #print(flow.shape)
            flow = resize(flow, roi_shape[::-1])
            #print(flow.shape)
        
        self.out.append(flow)

        self.prev_frame = cur  # update prev frame

        if len(self.out) == self.w_size:
            out = np.concatenate(self.out, axis=-1)
            self.out.pop(0)
            return out
        else:
            return None
