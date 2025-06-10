import numpy as np

# 原始数组
a = np.array([1, 2, 3])

# 用 concatenate 连接单数组（仅包含 a）
a_concatenated = np.concatenate([a])  # 输入是单数组

# 检查是否返回副本
print("a 的地址:", a.__array_interface__['data'][0])  # 原始数组的内存地址
print("a_concatenated 的地址:", a_concatenated.__array_interface__['data'][0])  # 新数组的内存地址

# 检查是否共享内存
print("是否共享内存:", np.may_share_memory(a, a_concatenated))  # 返回 False 说明不共享
print("a_concatenated 是 a 的副本吗?", a_concatenated.base is a)  # 返回 False 说明是副本