import numpy as np

def target_function():
    for _ in range(100000):
        arr = np.ndarray((1000, 1000))  # 直接构造 ndarray
        # pass
    return arr

# 使用 cProfile 分析
import cProfile
pr = cProfile.Profile()
pr.enable()
target_function()
pr.disable()

import pstats
ps = pstats.Stats(pr).sort_stats('tottime')
ps.print_stats()

ps.print_callees()