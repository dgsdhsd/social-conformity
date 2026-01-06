# -*- codeing = utf-8 -*-
# @Time : 2025/11/7 0:34
# @Author : 星空噩梦
# @File ： 21_眼动基础数据提取_一致性.py
# @Software : PyCharm
# -*- codeing = utf-8 -*-
# @Time : 2025/11/7 0:34
# @Author : 星空噩梦
# @File ： 21_眼动基础数据提取_一致性.py
# @Software : PyCharm

import os
import matplotlib
import pandas as pd
import numpy as np
import ast
# from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro

# ====================== 基本配置 ======================

Agentque = [2, 3, 1, 8, 7, 3, 6, 4, 8, 9, 4, 1, 9, 8, 7, 3, 2, 6, 2, 3]
TrueQue = [2, 1, 1, 8, 7, 3, 2, 4, 2, 9, 2, 1, 1, 8, 7, 3, 2, 4, 2, 9]
agentrespond = ['B', 'C', 'A', 'H', 'G', 'C', 'F', 'D', 'H', 'I',
                'D', 'A', 'I', 'H', 'G', 'C', 'B', 'F', 'B', 'C']

# trial_group 中的 trial 才进入一致性/非一致性统计
trial_group = [7, 18, 9, 20]

# 这些函数你原来就有，先保留
def remove_outliers(data):
    data = np.array(data)  # 确保数据是numpy数组
    lower_bound = np.percentile(data, 5)
    upper_bound = np.percentile(data, 95)
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data


def find_c_segments1(sequence, time_sequence):  # 这是整体的
    """
    Finds continuous segments in a sequence where all elements contain 'C'.
    Returns a list of tuples (start_index, end_index) for each segment.
    """
    segments = []
    start = None  # Track the start of a segment
    time = None
    pd_flag = 0

    for i, element in enumerate(sequence):
        if 'C' in element and pd_flag == 0:
            if start is None:  # Start of a new segment
                start = i
                pd_flag = 1

        elif 'C' not in element and pd_flag == 1:
            segments.append((start, i - 1))
            pd_flag = 0

    # Add the last segment if the sequence ends with a 'C' segment
    if start is not None:
        segments.append((start, len(sequence) - 1))

    return segments


# ====================== 主流程 ======================

path = os.getcwd()

# 回答数据路径
respondf = os.path.join(r"G:\myexperience\data\回答数据", "回答数据.xlsx")
respond_data = pd.read_excel(respondf)

need_four_square = [
    [9, 8, 4, 2], [7, 6, 3, 1], [8, 3, 9, 1], [2, 6, 4, 8],
    [8, 3, 2, 7], [3, 2, 1, 8], [2, 6, 8, 1], [4, 6, 1, 9],
    [2, 8, 4, 6], [9, 3, 7, 4]
]

# 存放每个 trial 的 raw fixation list（你之前的逻辑）
CONSISTENT = []
NONCONSISTENT = []

ceshi1=[]
ceshi2=[]

# 新增：存放每个 trial 的 saccade 平均速度
CONSISTENT_SACCADE_VEL = []
NONCONSISTENT_SACCADE_VEL = []

# ===== 遍历 1~30 号被试 =====
for ID in range(1, 31):
    print("ID:", ID)

    # 读取 fixation 结果（origin 版本）
    fix_file = os.path.join(r'G:\myexperience', str(ID),
                            'fixation_calculate_angle_0.8_time_100ms_origin.xlsx')
    if not os.path.exists(fix_file):
        print("  缺少 fixation 文件:", fix_file)
        continue

    # 读取 saccade 结果
    sacc_file = os.path.join(r'G:\myexperience', str(ID),
                             'saccade_calculate_angle_0.8_time_100ms.xlsx')
    if not os.path.exists(sacc_file):
        print("  缺少 saccade 文件:", sacc_file)
        continue

    fix_data = pd.read_excel(fix_file)
    sacc_data = pd.read_excel(sacc_file)

    # 每个被试 20 组 C1~C20
    for groupi in range(20):
        trial_index = groupi + 1
        if trial_index not in trial_group:
            continue

        # 当前 trial 对应的行
        # fixation_duration_simple 是一个列表的字符串，需要 ast 还原
        fix_simple_str = fix_data.loc[groupi, 'fixation_duration_simple']
        # 兼容空值或 NaN
        if isinstance(fix_simple_str, str):
            fix_simple = ast.literal_eval(fix_simple_str)
        elif isinstance(fix_simple_str, (list, tuple)):
            fix_simple = fix_simple_str
        else:
            # 比如是 NaN，视作没有数据
            fix_simple = []

        # 当前 trial 的 saccade 平均速度（直接用 saccade_velocity_average）
        sacc_vel_avg = sacc_data.loc[groupi, 'saccade_velocity_average']

        # 判断该 trial 是一致性还是非一致性
        if respond_data.iloc[ID - 1, 2 + groupi] == agentrespond[groupi]:
            # 一致性
            ceshi1.append(ID)
            CONSISTENT.append(fix_simple)
            CONSISTENT_SACCADE_VEL.append(sacc_vel_avg)
        else:
            # 非一致性
            ceshi2.append(ID)
            NONCONSISTENT.append(fix_simple)
            NONCONSISTENT_SACCADE_VEL.append(sacc_vel_avg)


# ====================== 统计指标计算 ======================

consistent_fixation_number = []
nonconsistent_fixation_number = []
consistent_fixation_average_duration = []
nonconsistent_fixation_average_duration = []
consistent_fixation_max = []
nonconsistent_fixation_max = []

# fixation 指标（一致性）
for simple in CONSISTENT:
    simple = np.array(simple, dtype=float)
    if simple.size == 0:
        # 如果没有数据，可以根据需要决定填 0 或 np.nan，这里先用 0/np.nan
        consistent_fixation_number.append(0)
        consistent_fixation_average_duration.append(np.nan)
        consistent_fixation_max.append(np.nan)
    else:
        consistent_fixation_number.append(len(simple))
        consistent_fixation_average_duration.append(simple.mean())
        consistent_fixation_max.append(simple.max())

# fixation 指标（非一致性）
for simple in NONCONSISTENT:
    simple = np.array(simple, dtype=float)
    if simple.size == 0:
        nonconsistent_fixation_number.append(0)
        nonconsistent_fixation_average_duration.append(np.nan)
        nonconsistent_fixation_max.append(np.nan)
    else:
        nonconsistent_fixation_number.append(len(simple))
        nonconsistent_fixation_average_duration.append(simple.mean())
        nonconsistent_fixation_max.append(simple.max())

# saccade 指标：这里直接用汇总好的平均速度列表
consistent_saccade_average_velocity = CONSISTENT_SACCADE_VEL
nonconsistent_saccade_average_velocity = NONCONSISTENT_SACCADE_VEL

print("一致性 trial 数:", len(CONSISTENT))
print("非一致性 trial 数:", len(NONCONSISTENT))
print("示例：一致性平均 saccade 速度前 5 个：", consistent_saccade_average_velocity[:5])
print("示例：非一致性平均 saccade 速度前 5 个：", nonconsistent_saccade_average_velocity[:5])
