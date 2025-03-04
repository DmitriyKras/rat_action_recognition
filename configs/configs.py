TOPVIEWRODENTS_SPINE_KPTS = [0, 3, 4, 7]
TOPVIEWRODENTS_SPINE_EARS_KPTS = [0, 1, 2, 3, 4, 7]
TOPVIEWRODENTS_SPINE_ANGLES = [[3, 0], [4, 3], [7, 4], [7, 0], [7, 3]]


TOPVIEWRODENTS_CONFIG = {'name': 'TopViewRodents', 'kpts': {'nose': 0, 'right_ear': 1, 'left_ear': 2, 'neck': 3, 'center': 4,
                                                            'right_side_body': 5, 'left_side_body': 6, 'tail_base': 7},
                        'root': '/home/cv-worker/dmitrii/RAT_DATASET/LAB_RAT_ACTIONS_DATASET', 
                        'videos': 'videos', 
                        'labels': 'labels_topviewrodents',
                        'optical_flow': 'optical_flow',
                        'classes': ['on_back_paws', 'body_cleaning', 'scratching_back_paw', 'grooming'],
                        'selected_ids': TOPVIEWRODENTS_SPINE_KPTS, 
                        'features': ['position', 'bbox', 'angles', 'distance', 'speed'],
                        #'features': ['position', 'distance'],
                        'angle_pairs': TOPVIEWRODENTS_SPINE_ANGLES,
                        'crop_features': False,
                        'img_features': 'img_features'}

W_SIZE = 16
