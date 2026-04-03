import numpy as np
import json
from pathlib import Path
base_path = Path(__file__).parent
correction_arr: np.ndarray = np.load(base_path / 'correction_results_D50_250805.npy', allow_pickle=True)
correction_info = correction_arr.item()
print(correction_info)
print(type(correction_info))
print(correction_info['wb_params'])
