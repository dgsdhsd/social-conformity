# -*- codeing = utf-8 -*-
# @Time : 2025/8/11 10:15
# @Author : 星空噩梦
# @File ： 2_时间序列图_静态AOI_Nback.py
# @Software : PyCharm

"""
计算静态 AOI 的 1-back 时间序列图,也可以计算4-back的时间序列图，但论文中并未体现
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==============================
# 配置区
# ==============================
Agentque = [2, 3, 1, 8, 7, 3, 6, 4, 8, 9, 4, 1, 9, 8, 7, 3, 2, 6, 2, 3]
TrueQue = [2, 1, 1, 8, 7, 3, 2, 4, 2, 9, 2, 1, 1, 8, 7, 3, 2, 4, 2, 9]

trial_group = [7, 18, 9, 20]
agentrespond = ['B', 'C', 'A', 'H', 'G', 'C', 'F', 'D', 'H', 'I',
                'D', 'A', 'I', 'H', 'G', 'C', 'B', 'F', 'B', 'C']

need_four_square = [
    [9, 8, 4, 2], [7, 6, 3, 1], [8, 3, 9, 1], [2, 6, 4, 8], [8, 3, 2, 7],
    [3, 2, 1, 8], [2, 6, 8, 1], [4, 6, 1, 9], [2, 8, 4, 6], [9, 3, 7, 4]
]

RESPOND_PATH = r"G:\myexperience\data\回答数据\回答数据.xlsx"
DATA_ROOT = r"G:\myexperience"

labels = ['ABC', 'DEF', 'GHI', 'Other']

# ==============================
# 函数区
# ==============================
def resize_nearest_row(row, target_cols):
    """对单一行进行最近邻插值，调整列数到目标值"""
    current_cols = len(row)
    if current_cols == target_cols:
        return row.copy()
    indices = np.linspace(0, current_cols - 1, num=target_cols)
    indices = np.round(indices).astype(int)
    return [row[i] for i in indices]


def resize_2d_keep_rows(arr, target_cols):
    """保持行数不变，调整每一行的列数到目标值"""
    return [resize_nearest_row(row, target_cols) for row in arr]


def convert_to_probability_matrix(consistent_resized):
    """
    将二维列表转换为概率矩阵：
    输入：58行 × N列
    输出：5行 × N列（每列概率表示 0~4 的比例）
    """
    arr = np.array(consistent_resized)
    n_rows, n_cols = arr.shape
    prob_matrix = np.zeros((5, n_cols), dtype=np.float64)

    for col in range(n_cols):
        counts = np.bincount(arr[:, col], minlength=5)
        prob_matrix[:, col] = counts / n_rows

    return prob_matrix.tolist()


def find_c_segments(sequence):
    """查找连续包含 'C' 的片段，返回 (start, end) 元组列表"""
    segments = []
    start = None
    for i, element in enumerate(sequence):
        if 'C' in element:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(sequence) - 1))
    return segments


# ==============================
# 主流程
# ==============================
NBACK_1 = []
NBACK_4 = []

respond_data = pd.read_excel(RESPOND_PATH)

for ID in range(1, 31):
    print(ID)

    fpath = os.path.join(DATA_ROOT, str(ID), 'fixation_angle_0.8_time_100ms.xlsx')
    data = pd.read_excel(fpath)

    state_list = data['state'].tolist()
    bzstate = find_c_segments(state_list)

    for groupi in range(20):
        AOI_time = []

        for i in range(bzstate[groupi][0], bzstate[groupi][1] + 1):
            time_diff = data['time'][i] - data['time'][i - 1]

            if data.at[i, 'gaze_x'] >= 3.2:
                if 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    AOI_time.extend([2] * time_diff)
                elif 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    AOI_time.extend([2] * time_diff)
                elif 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    AOI_time.extend([2] * time_diff)
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    AOI_time.extend([1] * time_diff)
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    AOI_time.extend([1] * time_diff)
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    AOI_time.extend([1] * time_diff)
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    AOI_time.extend([0] * time_diff)
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    AOI_time.extend([0] * time_diff)
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    AOI_time.extend([0] * time_diff)
                else:
                    AOI_time.extend([3] * time_diff)
            else:
                AOI_time.extend([3] * time_diff)

        if groupi + 1 in trial_group:
            NBACK_4.append(AOI_time)
        if groupi + 1 in [11, 2, 13]:
            NBACK_1.append(AOI_time)

# 重采样
NBACK_1_RESIZED = resize_2d_keep_rows(NBACK_1, target_cols=10000)
NBACK_4_RESIZED = resize_2d_keep_rows(NBACK_4, target_cols=10000)

# 概率矩阵
prob_NBACK_1 = convert_to_probability_matrix(NBACK_1_RESIZED)
prob_NBACK_4 = convert_to_probability_matrix(NBACK_4_RESIZED)

# 绘图
time = np.arange(10000)
plt.figure(figsize=(8, 6))
for i in range(4):
    smooth_data = gaussian_filter1d(prob_NBACK_1[i], sigma=10)
    plt.plot(time, smooth_data, label=labels[i], marker='o' if i == 0 else None)

plt.xlabel('Time since target onset (ms)')
plt.ylabel('Fixation probability')
plt.legend()
plt.show()
