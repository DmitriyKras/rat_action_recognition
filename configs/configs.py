TOPVIEWRODENTS_SPINE_KPTS = [0, 3, 4, 7]
TOPVIEWRODENTS_SPINE_ANGLES = [[3, 0], [4, 3], [7, 4], [7, 0], [7, 3]]

TOPVIEWRODENTS_SPINE_EARS_KPTS = [0, 1, 2, 3, 4, 7]
TOPVIEWRODENTS_SPINE_EARS_ANGLES = [[1, 0], [2, 0], [3, 0], [3, 1], [3, 2],  # head orientation
                                    [4, 3], [7, 4], [7, 0], [7, 3]]  # body orientation

TOPVIEWRODENTS_ALL_KPTS = [0, 1, 2, 3, 4, 5, 6, 7]
TOPVIEWRODENTS_ALL_ANGLES = [[1, 0], [2, 0], [3, 0], [3, 1], [3, 2],  # head orientation
                             [4, 3], [7, 0], [7, 3], [5, 3], [6, 3]  # body orientation
                             [7, 4], [7, 5], [7, 6]]  # hips orientation

TOPVIEWRODENTS_CONFIG = {'name': 'TopViewRodents', 'kpts': {'nose': 0, 'right_ear': 1, 'left_ear': 2, 'neck': 3, 'center': 4,
                                                            'right_side_body': 5, 'left_side_body': 6, 'tail_base': 7},
                        'root': '/home/cv-worker/dmitrii/RAT_DATASET/LAB_RAT_ACTIONS_DATASET', 
                        'videos': 'videos', 
                        'labels': 'labels_topviewrodents',
                        'optical_flow': 'optical_flow',
                        'kpts_features': 'kpts_features',
                        'classes': ['on_back_paws', 'body_cleaning', 'scratching_back_paw', 'grooming'],
                        'selected_ids': TOPVIEWRODENTS_SPINE_KPTS, 
                        'features': ['position', 'bbox', 'angles', 'distance', 'speed'],
                        #'features': ['position', 'distance'],
                        'angle_pairs': TOPVIEWRODENTS_SPINE_ANGLES}

W_SIZE = 16

WISTAR_RAT_SPINE_KPTS = [0, 3, 6, 7, 10]
WISTAR_RAT_SPINE_ANGLES = [[3, 0], [6, 3], [7, 6], [10, 7],  # spine orientation
                           [10, 3], [10, 0]]

WISTAR_RAT_SPINE_HEAD_KPTS = [0, 1, 2, 3, 4, 5, 6, 7, 10]
WISTAR_RAT_SPINE_HEAD_ANGLES = [[3, 0], [6, 3], [7, 6], [10, 7],  # spine orientation
                                [4, 0], [5, 0],  # ears orientation 
                                [10, 3],  # body orientation
                                [10, 0]]  # rat orientation

WISTAR_RAT_ALL_KPTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
WISTAR_RAT_ALL_ANGLES = [[3, 0], [6, 3], [7, 6], [10, 7],  # spine orientation
                        [4, 0], [5, 0],  # ears orientation 
                        [10, 3],  # body orientation
                        [10, 11], [10, 12], [11, 8], [12, 9],  # hips orientation
                        [6, 8], [6, 9], # shoulders orientation
                        [10, 0]]  # rat orientation

WISTAR_RAT_CONFIG = {'name': 'WistarRat', 'kpts': {'nose': 0, 'right_eye': 1, 'left_eye': 2, 'head': 3, 
                                                   'right_ear': 4, 'left_ear': 5, 'shoulders_center': 6,
                                                   'body_center': 7, 'right_shoulder': 8, 'left_shoulder': 9,
                                                   'tail_base': 10, 'right_hip': 11, 'left_hip': 12},
                        'root': '/home/cv-worker/dmitrii/RAT_DATASET/LAB_RAT_ACTIONS_DATASET', 
                        'videos': 'videos', 
                        'labels': 'labels_ratpose',
                        'optical_flow': 'optical_flow',
                        'kpts_features': 'kpts_features',
                        'classes': ['on_back_paws', 'body_cleaning', 'scratching_back_paw', 'grooming'],
                        'selected_ids': WISTAR_RAT_SPINE_KPTS, 
                        'features': ['position', 'bbox', 'angles', 'distance', 'speed'],
                        #'features': ['position', 'distance'],
                        'angle_pairs': WISTAR_RAT_SPINE_ANGLES}
