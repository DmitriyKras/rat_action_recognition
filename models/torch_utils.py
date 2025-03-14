from typing import Dict


def state_dict_intersection(dict1: Dict, dict2: Dict) -> Dict:
    pretrained_dict = {k: v for k, v in dict2.items() if k in dict1}
    dict1.update(pretrained_dict)
    return dict1
