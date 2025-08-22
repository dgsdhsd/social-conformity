# -*- codeing = utf-8 -*-
# @Time : 2025/4/2 14:59
# @Author : 星空噩梦
# @File ： 8_视线方向角度_时间序列图.py
# @Software : PyCharm

'''
angle具体获取来源为angle3.excel和angle4.excel，通过计算其平均值。
该代码主要为了绘制视线方向角度的时间序列图
'''
import os
import matplotlib
import pandas as pd
import numpy as np
#from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# 9个时间点上的两个序列：consistent 和 non-consistent
angle = [[9.81, 2.91], [15.93, 6.55], [18.99, 10.76], [18.12, 13.23],
         [12.47, 22.74], [17.63, 27.81], [18.53, 25.01], [23.64, 25.86],
         [36.78, 34.66]]

# 让第一个点从 x=1 开始，最后一个点在 x=9
time = np.linspace(1, 9, len(angle))

# 提取两条序列
consistent = [a[0] for a in angle]
non_consistent = [a[1] for a in angle]

# 高斯平滑
sigma = 2  # 适当的平滑度
consistent_smooth=consistent
non_consistent_smooth=non_consistent
# consistent_smooth = gaussian_filter1d(consistent, sigma=sigma)
# non_consistent_smooth = gaussian_filter1d(non_consistent, sigma=sigma)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(time, consistent_smooth, label="consistent", marker='o', linestyle='-', color='b')
plt.plot(time, non_consistent_smooth, label="non-consistent", marker='s', linestyle='--', color='r')

# 设置 x 轴范围 0-10
plt.xlim(0, 10)

# 轴标签
plt.xlabel('Time since target onset (ms)')
plt.ylabel('Gaze direction angle')

# 图例
plt.legend()

# 显示图形
plt.show()