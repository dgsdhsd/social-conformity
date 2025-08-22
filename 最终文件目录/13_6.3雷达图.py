# -*- codeing = utf-8 -*-
# @Time : 2025/6/3 14:00
# @Author : 星空噩梦
# @File ： 13_6.3雷达图.py
# @Software : PyCharm


import os
import matplotlib
import pandas as pd
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体显示中文

# AOI 标签和数据
aoi_labels = ['L4', 'L3', 'L2', 'L1', 'Other']

data_1_3_consistent=[0.0944102239364458, 0.18648897587933913, 0.01650451902596281, 0.11744286454435554, 0.5851534166138967]

data_1_3_nonconsistent=[0.14788276078058685, 0.1704482866615549, 0.02305690249144141, 0.08005658930634901, 0.5785554607600678]

data_3_6_consistent = [0.027750963790630127, 0.0587548449912684, 0.001860243879124965, 0.0496247911504295, 0.8620091561885468]
data_3_6_nonconsistent = [0.09229906832044427, 0.1506870136939776, 0.014315464833601505, 0.052816195976190645, 0.6898822571757861]

data1=data_1_3_consistent
data2=data_3_6_consistent
# 闭合数据以绘制雷达图（首尾相连）
data1 += data1[:1]
data2 += data2[:1]
angles = np.linspace(0, 2 * np.pi, len(aoi_labels), endpoint=False).tolist()
angles += angles[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 绘制 data1
ax.plot(angles, data1, color='blue', linewidth=2, label='0-3秒时间窗')
ax.fill(angles, data1, color='blue', alpha=0.25)

# 绘制 data2
ax.plot(angles, data2, color='red', linewidth=2, label='3-6秒时间窗')
ax.fill(angles, data2, color='red', alpha=0.25)

# 设置坐标标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(aoi_labels)

# 其他美化设置
ax.set_title("一致性行为不同时间窗雷达图", size=14, pad=20)
ax.set_rlabel_position(30)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.show()




data1=data_1_3_nonconsistent
data2=data_3_6_nonconsistent
# 闭合数据以绘制雷达图（首尾相连）
data1 += data1[:1]
data2 += data2[:1]
angles = np.linspace(0, 2 * np.pi, len(aoi_labels), endpoint=False).tolist()
angles += angles[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 绘制 data1
ax.plot(angles, data1, color='blue', linewidth=2, label='0-3秒时间窗')
ax.fill(angles, data1, color='blue', alpha=0.25)

# 绘制 data2
ax.plot(angles, data2, color='red', linewidth=2, label='3-6秒时间窗')
ax.fill(angles, data2, color='red', alpha=0.25)

# 设置坐标标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(aoi_labels)

# 其他美化设置
ax.set_title("非一致性行为不同时间窗雷达图", size=14, pad=20)
ax.set_rlabel_position(30)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.show()