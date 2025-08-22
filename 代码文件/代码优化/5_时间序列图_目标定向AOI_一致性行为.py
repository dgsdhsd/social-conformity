# -*- codeing = utf-8 -*-
# @Time : 2025/8/11 14:13
# @Author : 星空噩梦
# @File ： 5_时间序列图_目标定向AOI_一致性行为.py
# @Software : PyCharm

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 代理队列与真实队列，顺序对应
agent_queue = [2, 3, 1, 8, 7, 3, 6, 4, 8, 9, 4, 1, 9, 8, 7, 3, 2, 6, 2, 3]
true_queue = [2, 1, 1, 8, 7, 3, 2, 4, 2, 9, 2, 1, 1, 8, 7, 3, 2, 4, 2, 9]

def resize_row_nearest(row, target_cols):
    """
    最近邻插值调整单行数据长度至目标列数
    """
    current_cols = len(row)
    if current_cols == target_cols:
        return row.copy()
    indices = np.linspace(0, current_cols - 1, num=target_cols)
    indices = np.round(indices).astype(int)
    return [row[i] for i in indices]

def resize_2d_rows_keep(arr, target_cols):
    """
    保持二维列表行数，调整每行列数至目标值
    """
    return [resize_row_nearest(row, target_cols) for row in arr]

def to_probability_matrix(data_2d):
    """
    将二维列表(58x10000)转换成概率矩阵(5x10000)，
    统计每列中0~4出现的比例
    """
    arr = np.array(data_2d)
    n_rows, n_cols = arr.shape
    prob_mat = np.zeros((5, n_cols), dtype=np.float64)
    for col in range(n_cols):
        counts = np.bincount(arr[:, col], minlength=5)
        prob_mat[:, col] = counts / n_rows
    return prob_mat.tolist()

def find_c_segments(sequence):
    """
    查找序列中连续包含字符'C'的段落，返回起止索引元组列表
    """
    segments = []
    start = None
    for i, val in enumerate(sequence):
        if 'C' in val:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(sequence) - 1))
    return segments

# 目录与数据路径
path = os.getcwd()
need_four_square = [
    [9, 8, 4, 2], [7, 6, 3, 1], [8, 3, 9, 1], [2, 6, 4, 8],
    [8, 3, 2, 7], [3, 2, 1, 8], [2, 6, 8, 1], [4, 6, 1, 9],
    [2, 8, 4, 6], [9, 3, 7, 4]
]

respond_file = os.path.join("G:\\myexperience\\data\\回答数据", "回答数据.xlsx")
respond_data = pd.read_excel(respond_file)

consistent = []
non_consistent = []
number = 0

trial_group = [7, 18, 9, 20]
agent_respond = ['B', 'C', 'A', 'H', 'G', 'C', 'F', 'D', 'H', 'I', 'D', 'A', 'I', 'H', 'G', 'C', 'B', 'F', 'B', 'C']

for subject_id in range(1, 31):
    print(subject_id)
    file_path = os.path.join("G:\\myexperience", str(subject_id), 'fixation_angle_0.8_time_100ms.xlsx')
    data = pd.read_excel(file_path)
    state_list = data['state'].tolist()
    c_segments = find_c_segments(state_list)
    for group_idx in range(20):
        lists = [0] * 14
        aoi_time = []
        segment_start, segment_end = c_segments[group_idx]
        four_square = need_four_square[group_idx % 10]
        for i in range(segment_start, segment_end + 1):
            time = data['time'][i] - data['time'][i - 1]
            gaze_x = data.at[i, 'gaze_x']
            gaze_y = data.at[i, 'gaze_y']
            gaze_z = data.at[i, 'gaze_z']
            if gaze_x >= 3.2:
                pdd = 0
                if 0.525 <= gaze_y <= 0.905:
                    if -7.345 <= gaze_z <= -6.965:
                        lists[8] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 8:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                    elif -6.965 <= gaze_z <= -6.585:
                        lists[7] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 7:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                    elif -6.585 <= gaze_z <= -6.205:
                        lists[6] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 6:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                elif 0.905 <= gaze_y <= 1.285:
                    if -7.345 <= gaze_z <= -6.965:
                        lists[5] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 5:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                    elif -6.965 <= gaze_z <= -6.585:
                        lists[4] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 4:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                    elif -6.585 <= gaze_z <= -6.205:
                        lists[3] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 3:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                elif 1.285 <= gaze_y <= 1.665:
                    if -7.345 <= gaze_z <= -6.965:
                        lists[2] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 2:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                    elif -6.965 <= gaze_z <= -6.585:
                        lists[1] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 1:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                    elif -6.585 <= gaze_z <= -6.205:
                        lists[0] += time
                        for idx in range(4):
                            if four_square[idx] - 1 == 0:
                                aoi_time.extend([idx] * time)
                                pdd = 1
                                break
                if pdd == 0:
                    lists[9] += time
                    aoi_time.extend([4] * time)
            else:
                lists[9] += time
                aoi_time.extend([4] * time)
                continue
        if group_idx + 1 in trial_group:
            if respond_data.iloc[subject_id - 1, 2 + group_idx] == agent_respond[group_idx]:
                consistent.append(aoi_time)
                #print(number)
                number += 1
            else:
                non_consistent.append(aoi_time)
                #print(number)
                number += 1

# 调整数据长度到10000列
consistent_resized = resize_2d_rows_keep(consistent, target_cols=10000)
non_consistent_resized = resize_2d_rows_keep(non_consistent, target_cols=10000)

prob_consistent = to_probability_matrix(consistent_resized)
prob_non_consistent = to_probability_matrix(non_consistent_resized)

time = np.arange(10000)
labels = ['4back', '3back', '2back', '1back', 'Other']

plt.figure(figsize=(8, 6))
for i in range(5):
    smooth_data = gaussian_filter1d(prob_non_consistent[i], sigma=10)
    plt.plot(time, smooth_data, label=labels[i], marker='o' if i == 0 else None)

plt.xlabel('Time since target onset (ms)')
plt.ylabel('Fixation probability')
plt.legend()
plt.show()
