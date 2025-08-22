# -*- codeing = utf-8 -*-
# @Time : 2025/8/11 9:47
# @Author : 星空噩梦
# @File ： 1_时间序列图_目标定向AOI_Nback.py
# @Software : PyCharm

"""
该脚本用于生成 4back 与 1back 的时间序列图。
如需绘制 4back图，只需在    smooth_data = gaussian_filter1d(prob_NBACK_1[i], sigma=10)将
    prob_NBACK_1[i]
替换为
    prob_NBACK_4[i]
并重新运行。

数据说明：
- 输入：眼动数据（fixation_angle_0.8_time_100ms.xlsx）、回答数据（回答数据.xlsx）
- 输出：平滑处理后的注视概率时间序列图
"""

import os
import matplotlib
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 使用 TkAgg 后端

# ===================== 常量定义 =====================
AGENT_SEQUENCE = [2, 3, 1, 8, 7, 3, 6, 4, 8, 9, 4, 1, 9, 8, 7, 3, 2, 6, 2, 3]
TRUE_SEQUENCE = [2, 1, 1, 8, 7, 3, 2, 4, 2, 9, 2, 1, 1, 8, 7, 3, 2, 4, 2, 9]

TRIAL_GROUP_4BACK = [7, 18, 9, 20]  # 属于 4back 的 trial index（1-based）
TRIAL_GROUP_1BACK = [11, 2, 13]     # 属于 1back 的 trial index（1-based）

AGENT_RESPOND = ['B', 'C', 'A', 'H', 'G', 'C', 'F', 'D', 'H', 'I',
                 'D', 'A', 'I', 'H', 'G', 'C', 'B', 'F', 'B', 'C']

NEED_FOUR_SQUARE = [
    [9, 8, 4, 2], [7, 6, 3, 1], [8, 3, 9, 1], [2, 6, 4, 8], [8, 3, 2, 7],
    [3, 2, 1, 8], [2, 6, 8, 1], [4, 6, 1, 9], [2, 8, 4, 6], [9, 3, 7, 4]
]

# ===================== 工具函数 =====================
def resize_nearest_row(row, target_cols):
    """对单行数据进行最近邻插值，使列数调整到目标值"""
    current_cols = len(row)
    if current_cols == target_cols:
        return row.copy()
    indices = np.linspace(0, current_cols - 1, num=target_cols)
    indices = np.round(indices).astype(int)
    return [row[i] for i in indices]

def resize_2d_keep_rows(arr, target_cols):
    """保持行数不变，对二维列表每行进行最近邻插值"""
    return [resize_nearest_row(row, target_cols) for row in arr]

def convert_to_probability_matrix(data_2d):
    """
    将 N行×M列 的二维列表转换为 5行×M列 的概率矩阵。
    每列表示该时间点 0~4 这 5 类的出现概率。
    """
    arr = np.array(data_2d)
    n_rows, n_cols = arr.shape
    prob_matrix = np.zeros((5, n_cols), dtype=np.float64)

    for col in range(n_cols):
        counts = np.bincount(arr[:, col], minlength=5)
        prob_matrix[:, col] = counts / n_rows

    return prob_matrix.tolist()

def find_c_segments(sequence):
    """
    找出序列中连续包含 'C' 的片段。
    返回：[(start_idx, end_idx), ...]
    """
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

# ===================== 数据读取 =====================
response_path = os.path.join(r"G:\myexperience\data\回答数据", "回答数据.xlsx")
response_df = pd.read_excel(response_path)

NBACK_1, NBACK_4 = [], []

for participant_id in range(1, 31):
    print(f"Processing participant {participant_id} ...")
    fixation_path = os.path.join(r"G:\myexperience", str(participant_id), "fixation_angle_0.8_time_100ms.xlsx")
    fixation_df = pd.read_excel(fixation_path)

    for trial_idx in range(20):
        time_list = [0] * 14
        AOI_time = []

        state_list = fixation_df['state'].tolist()
        c_segments = find_c_segments(state_list)

        for i in range(c_segments[trial_idx][0], c_segments[trial_idx][1] + 1):
            time_diff = fixation_df['time'][i] - fixation_df['time'][i - 1]
            four_square = NEED_FOUR_SQUARE[trial_idx % 10]

            # gaze_x >= 3.2 才进入 AOI 判断
            if fixation_df.at[i, 'gaze_x'] >= 3.2:
                gaze_y = fixation_df.at[i, 'gaze_y']
                gaze_z = fixation_df.at[i, 'gaze_z']
                # 区域判断（共 9 个 AOI + 其他）
                if 0.525 <= gaze_y <= 0.905 and -7.345 <= gaze_z <= -6.965:
                    time_list[8] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 8:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                elif 0.525 <= gaze_y <= 0.905 and -6.965 <= gaze_z <= -6.585:
                    time_list[7] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 7:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                elif 0.525 <= gaze_y <= 0.905 and -6.585 <= gaze_z <= -6.205:
                    time_list[6] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 6:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                elif 0.905 <= gaze_y <= 1.285 and -7.345 <= gaze_z <= -6.965:
                    time_list[5] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 5:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                elif 0.905 <= gaze_y <= 1.285 and -6.965 <= gaze_z <= -6.585:
                    time_list[4] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 4:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                elif 0.905 <= gaze_y <= 1.285 and -6.585 <= gaze_z <= -6.205:
                    time_list[3] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 3:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                elif 1.285 <= gaze_y <= 1.665 and -7.345 <= gaze_z <= -6.965:
                    time_list[2] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 2:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                elif 1.285 <= gaze_y <= 1.665 and -6.965 <= gaze_z <= -6.585:
                    time_list[1] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 1:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                elif 1.285 <= gaze_y <= 1.665 and -6.585 <= gaze_z <= -6.205:
                    time_list[0] += time_diff
                    pdd_flag = 0
                    for group_i in range(4):
                        if four_square[group_i] - 1 == 0:
                            AOI_time.extend([group_i] * time_diff)
                            pdd_flag = 1
                            break
                    if pdd_flag == 0:
                        AOI_time.extend([4] * time_diff)

                else:
                    time_list[9] += time_diff
                    AOI_time.extend([4] * time_diff)

            else:
                time_list[9] += time_diff
                AOI_time.extend([4] * time_diff)

        if trial_idx + 1 in TRIAL_GROUP_4BACK:
            NBACK_4.append(AOI_time)
        if trial_idx + 1 in TRIAL_GROUP_1BACK:
            NBACK_1.append(AOI_time)

# ===================== 概率计算 =====================
NBACK_1_RESIZED = resize_2d_keep_rows(NBACK_1, target_cols=10000)
NBACK_4_RESIZED = resize_2d_keep_rows(NBACK_4, target_cols=10000)

prob_NBACK_1 = convert_to_probability_matrix(NBACK_1_RESIZED)
prob_NBACK_4 = convert_to_probability_matrix(NBACK_4_RESIZED)

# ===================== 绘图 =====================
time_axis = np.arange(10000)
labels = ['4back', '3back', '2back', '1back', 'Other']

plt.figure(figsize=(8, 6))
for i in range(5):
    smooth_data = gaussian_filter1d(prob_NBACK_4[i], sigma=10)  # 可切换 prob_NONCONSISTENT
    plt.plot(time_axis, smooth_data, label=labels[i], marker='o' if i == 0 else None)

plt.xlabel('Time since target onset (ms)')
plt.ylabel('Fixation probability')
plt.legend()
plt.show()
