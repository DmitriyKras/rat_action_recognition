import torch
from typing import Dict, Literal
import numpy as np
from numpy import random
PI = torch.tensor(np.pi)


class KeypointsAugmentor:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.n_kpts = len(config['selected_ids'])  # number of kpts in feature vector
        self.features = config['features']
        self.n_angles = len(config['angle_pairs'])

    def __flip_position(self, pos_features: torch.Tensor, 
                        type: Literal['horizontal', 'vertical']) -> torch.Tensor:
        assert pos_features.size()[1] == self.n_kpts * 2, f"Number of kpts is {self.n_kpts}, but size of pos features is {pos_features.size()}"
        w_size = pos_features.size()[0]
        kpts = pos_features.reshape(w_size, -1, 2)
        if type == 'horizontal':
            kpts[:, :, 0] = 1 - kpts[:, :, 0]
        elif type == 'vertical':
            kpts[:, :, 1] = 1 - kpts[:, :, 1]
        return kpts.reshape(w_size, -1)
    
    def __flip_angles(self, angle_features: torch.Tensor, 
                        type: Literal['horizontal', 'vertical']) -> torch.Tensor:
        assert angle_features.size()[1] == self.n_angles, f"Number of angles is {self.n_angles}, but size of angle features is {angle_features.size()}"
        if type == 'horizontal':
            angle_features = torch.remainder(3 * PI - angle_features, PI)
        elif type == 'vertical':
            angle_features = 2 * PI - angle_features
        return angle_features
    
    def __flip_speed_acceleration(self, speed_features: torch.Tensor, 
                     type: Literal['horizontal', 'vertical']) -> torch.Tensor:
        assert speed_features.size()[1] == self.n_kpts * 2, f"Number of kpts is {self.n_kpts}, but size of speed features is {speed_features.size()}"
        w_size = speed_features.size()[0]
        speed_xy = speed_features.reshape(w_size, -1, 2)
        if type == 'horizontal':
            speed_xy[:, :, 0] *= -1
        elif type == 'vertical':
            speed_xy[:, :, 1] *= -1
        return speed_xy.reshape(w_size, -1)
    
    def __flip(self, f: torch.Tensor, type: Literal['horizontal', 'vertical']) -> torch.Tensor:
        # Perform flip of all feature vectors sequence
        ind = 0  # index of current position in f vector
        out_features = []
        if 'position' in self.features:
            out_features.append(self.__flip_position(f[:, :self.n_kpts * 2], type))
            ind += self.n_kpts * 2
        if 'bbox' in self.features:
            out_features.append(f[:, ind : ind + 3])
            ind += 3
        if 'angles' in self.features:
            out_features.append(self.__flip_angles(f[:, ind : ind + self.n_angles], type))
            ind += self.n_angles
        if 'distance' in self.features:
            n = self.n_kpts * (self.n_kpts - 1) // 2
            out_features.append(f[:, ind : ind + n])
            ind += n
        if 'speed' in self.features:
            out_features.append(self.__flip_speed_acceleration(f[:, ind : ind + self.n_kpts * 2], type))
            ind += self.n_kpts * 2
        if 'acceleration' in self.features:
            out_features.append(self.__flip_speed_acceleration(f[:, ind : ind + self.n_kpts * 2], type))
            ind += self.n_kpts * 2
        return torch.cat(out_features, dim=-1)
    
    def augment(self, f: torch.Tensor, p: float = 0.2) -> torch.Tensor:
        # Perform augmentation with p probability
        # Horizontal
        prob = random.rand()
        # print(f"Initial shape {f.size()}")
        if prob < p:
            f = self.__flip(f, 'horizontal')
        # Vertical
        # print(f"After horizontal shape {f.size()}")
        prob = random.rand()
        if prob < p:
            f = self.__flip(f, 'vertical')
        # print(f"After vertical shape {f.size()}")
        return f
        