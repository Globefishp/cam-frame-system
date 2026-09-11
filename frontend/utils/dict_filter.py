# backend/utils/dict_filter.py
# Author: Gemini 3.1 pro
# Recursively filter and update dict.

import copy
from typing import Optional, Any, Tuple, Dict, List

def filter_dict(d: dict, paths: Optional[List[str]]) -> dict:
    """
    Extract a sub-dictionary from a nested dictionary based on a list of paths.
    
    :param d: The source dictionary to extract from.
    :param paths: A list of string paths, e.g. ["camera.exposure_time_ms", "piezo.target_pos"].
                  If None or empty, returns a deep copy of the original dictionary.
    :return: A new dictionary containing only the specified paths and their values.
    """
    if not paths:
        return copy.deepcopy(d)
    
    result = {}
    for path in paths:
        parts = path.split('.')
        current_d = d
        current_res = result
        valid = True
        
        for i, part in enumerate(parts):
            if part not in current_d:
                valid = False
                break
            
            if i == len(parts) - 1:
                current_res[part] = copy.deepcopy(current_d[part])
            else:
                if part not in current_res:
                    current_res[part] = {}
                current_res = current_res[part]
                current_d = current_d[part]
                if not isinstance(current_d, dict):
                    valid = False
                    break
    return result

def rupdate_dict(d: dict, u: dict) -> dict:
    """
    Recursively update a nested dictionary with another dictionary.
    
    :param d: The original dictionary to be updated (modified in-place).
    :param u: The dictionary containing new values to merge into `d`.
    :return: The updated dictionary `d`.
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            rupdate_dict(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d
