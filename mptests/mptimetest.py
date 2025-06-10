import time

info = time.get_clock_info('time')
print(f"理论精度: {info.resolution * 1e6:.2f} 微秒")
print(f"实现方式: {info.implementation}")
print(f"可调整: {info.adjustable}")


import time
import numpy as np

def measure_time_precision():
    # 测量连续调用间的最小间隔
    min_diff = float('inf')
    zero_count = 0
    equal_count = 0
    total = 100000
    
    for _ in range(total):
        t1 = time.time()
        t2 = time.time()
        diff = t2 - t1
        
        if diff > 0:
            min_diff = min(min_diff, diff)
        elif diff == 0:
            zero_count += 1
        else:  # 理论上不应发生
            print(f"负差值! {diff}")
    
    # 分析相同时间戳的频率
    same_rate = zero_count / total
    
    print(f"\n=== 测试结果 ===")
    print(f"最小正差值: {min_diff * 1e6:.2f} 微秒")
    print(f"相同时间戳比例: {same_rate * 100:.2f}%")
    # print(f"估计实际精度: {min_diff * 1e3:.2f} 毫秒")

if __name__ == "__main__":
    measure_time_precision()