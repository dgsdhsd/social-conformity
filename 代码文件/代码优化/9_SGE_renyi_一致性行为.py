# -*- codeing = utf-8 -*-
# @Time : 2025/8/11 15:15
# @Author : 星空噩梦
# @File ： 9_SGE_renyi_一致性行为.py
# @Software : PyCharm

"""
主要计算一致性行为和非一致性行为的 SGE
"""

import os
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# matplotlib 后端
matplotlib.use('TkAgg')

# 数据与实验参数
Agentque = [2, 3, 1, 8, 7, 3, 6, 4, 8, 9, 4, 1, 9, 8, 7, 3, 2, 6, 2, 3]
TrueQue = [2, 1, 1, 8, 7, 3, 2, 4, 2, 9, 2, 1, 1, 8, 7, 3, 2, 4, 2, 9]
agentrespond = ['B', 'C', 'A', 'H', 'G', 'C', 'F', 'D', 'H', 'I',
                'D', 'A', 'I', 'H', 'G', 'C', 'B', 'F', 'B', 'C']

CONSISTENT = []
NONCONSISTENT = []
trial_group = [7, 18, 9, 20]

# 工具函数
def remove_outliers(data):
    """去除数据中的 5% 和 95% 分位之外的值"""
    data = np.array(data)
    lower_bound = np.percentile(data, 5)
    upper_bound = np.percentile(data, 95)
    return data[(data >= lower_bound) & (data <= upper_bound)]


def calculate_entropy(values):
    """计算 Shannon 熵（log2）"""
    values = [v for v in values if v != 0]
    if not values:
        print("出现问题时 values:", values)
        return 0.0

    total = sum(values)
    if total == 0:
        print("出现问题时 values:", values)
        return 0.0

    probabilities = [v / total for v in values]
    entropy = -sum(p * np.log2(p) for p in probabilities if p != 0)

    if np.isnan(entropy):
        print("出现问题时 values:", values)
        print("计算得到的概率:", probabilities)

    return entropy


def generalized_entropy(counts, alpha):
    """
    计算广义熵（Rényi 熵, log2 基础）
    alpha = 1 → Shannon 熵
    """
    counts = np.array(counts, dtype=np.float64)

    if np.any(counts < 0):
        raise ValueError("Counts must be non-negative")

    total = np.sum(counts)
    if total == 0:
        raise ValueError("Total count cannot be zero")

    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]

    if alpha <= 0:
        raise ValueError("Alpha must be greater than 0")

    if alpha == 1:
        entropy = -np.sum(probabilities * np.log2(probabilities))
    else:
        entropy = (1 / (1 - alpha)) * np.log2(np.sum(probabilities ** alpha))

    return entropy


# 寻找特定 'C' 段落的三种方法
def find_c_segments1(sequence, time_sequence):
    segments = []
    start = None
    time = None
    pd_state = 0

    for i, element in enumerate(sequence):
        if 'C' in element and pd_state == 0:
            if start is None:
                start = i
                time = time_sequence[i] + 6000
                pd_state = 1
        elif 'C' in element and pd_state == 1:
            if time_sequence[i] > time:
                segments.append((start, i - 1))
                start = None
                pd_state = 2
        elif 'C' not in element and pd_state == 2:
            pd_state = 0

    if start is not None:
        segments.append((start, len(sequence) - 1))

    return segments


def find_c_segments2(sequence, time_sequence):
    segments = []
    start = None
    true_start = None
    time_first = None
    time_last = None
    pd_state = 0

    for i, element in enumerate(sequence):
        if 'C' in element and pd_state == 0:
            if start is None:
                start = i
                time_first = time_sequence[i] + 4000
                time_last = time_first + 3000
            elif true_start is None and time_sequence[i] > time_first:
                true_start = i
            elif time_sequence[i] > time_last:
                segments.append((true_start, i - 1))
                if time_sequence[i - 1] - time_sequence[true_start] > 5000:
                    print(start, true_start, time_first, time_last, pd_state)
                start = None
                true_start = None
                time_first = None
                time_last = None
                pd_state = 1
        else:
            pd_state = 0
            start = None
            true_start = None
            time_first = None
            time_last = None

    if start is not None:
        segments.append((start, len(sequence) - 1))

    return segments


def find_c_segments3(sequence, time_sequence):
    segments = []
    start = None
    time = 0
    true_start = None

    for i, element in enumerate(sequence):
        if 'C' in element:
            if start is None:
                start = i
                time = time_sequence[i] + 7000
            elif time_sequence[i] > time and true_start is None:
                true_start = i
        else:
            if start is not None:
                segments.append((true_start, i - 1))
                start = None
                true_start = None

    if true_start is not None:
        segments.append((true_start, len(sequence) - 1))

    return segments


# 数据路径
path = os.getcwd()
respondf = os.path.join(r"G:\myexperience\data\回答数据", "回答数据.xlsx")
respond_data = pd.read_excel(respondf)

# 四宫格对应关系
need_four_square = [
    [9, 8, 4, 2], [7, 6, 3, 1], [8, 3, 9, 1], [2, 6, 4, 8], [8, 3, 2, 7],
    [3, 2, 1, 8], [2, 6, 8, 1], [4, 6, 1, 9], [2, 8, 4, 6], [9, 3, 7, 4]
]

# 主循环
for ID in range(1, 31):
    print(ID)
    LSATLIST = []
    f = os.path.join(r'G:\myexperience', str(ID), 'fixation_angle_0.8_time_100ms.xlsx')
    data = pd.read_excel(f)

    for groupi in range(20):
        lists = [0] * 14
        state_list = data['state'].tolist()
        time_list = data['time'].tolist()

        bzstate = find_c_segments1(state_list, time_list)
        # bzstate = find_c_segments2(state_list, time_list)
        # bzstate = find_c_segments3(state_list, time_list)

        for i in range(bzstate[groupi][0], bzstate[groupi][1] + 1):
            time_diff = data['time'][i] - data['time'][i - 1]
            if data.at[i, 'gaze_x'] >= 3.2:
                if 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[8] += time_diff
                elif 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[7] += time_diff
                elif 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[6] += time_diff
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[5] += time_diff
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[4] += time_diff
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[3] += time_diff
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[2] += time_diff
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[1] += time_diff
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[0] += time_diff
                else:
                    lists[9] += time_diff
            else:
                lists[9] += time_diff
                continue

        four_square = need_four_square[groupi % 10]
        lists[10] = lists[four_square[0] - 1]
        lists[11] = lists[four_square[1] - 1]
        lists[12] = lists[four_square[2] - 1]
        lists[13] = lists[four_square[3] - 1]

        lists.append(
            sum(lists[:10]) - lists[10] - lists[11] - lists[12] - lists[13]
        )

        LSATLIST.append(lists)

        if groupi + 1 in trial_group:
            if respond_data.iloc[ID - 1, 2 + groupi] == agentrespond[groupi]:
                CONSISTENT.append(lists)
            else:
                NONCONSISTENT.append(lists)

# 计算广义熵并进行 T 检验
alphas = [0.5, 0.7, 1, 2, 3]

for alpha in alphas:
    entropies_CONSISTENT = [
        generalized_entropy(row[-5:], alpha) for row in CONSISTENT
    ]
    entropies_NONCONSISTENT = [
        generalized_entropy(row[-5:], alpha) for row in NONCONSISTENT
    ]

    mean_consistent = np.mean(entropies_CONSISTENT)
    mean_nonconsistent = np.mean(entropies_NONCONSISTENT)
    std_consistent = np.std(entropies_CONSISTENT)
    std_nonconsistent = np.std(entropies_NONCONSISTENT)
    offset = mean_consistent - mean_nonconsistent

    t_statistic, p_value_t = ttest_ind(entropies_CONSISTENT, entropies_NONCONSISTENT)

    print(f"\n=== α = {alpha} ===")
    print("CONSISTENT 组均值:", mean_consistent)
    print("NONCONSISTENT 组均值:", mean_nonconsistent)
    print("CONSISTENT 组标准差:", std_consistent)
    print("NONCONSISTENT 组标准差:", std_nonconsistent)
    print("偏移值 (CONSISTENT - NONCONSISTENT):", offset)
    print("T 检验结果:")
    print("t 统计量:", t_statistic)
    print("p 值:", p_value_t)
