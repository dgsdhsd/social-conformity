# -*- codeing = utf-8 -*-
# @Time : 2025/11/7 0:34
# @Author : 星空噩梦
# @File ： 22_眼动基础数据提取_Nback.py
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

# ====================== 配置区 ======================

Agentque = [2, 3, 1, 8, 7, 3, 6, 4, 8, 9, 4, 1, 9, 8, 7, 3, 2, 6, 2, 3]
TrueQue = [2, 1, 1, 8, 7, 3, 2, 4, 2, 9, 2, 1, 1, 8, 7, 3, 2, 4, 2, 9]
agentrespond = ['B', 'C', 'A', 'H', 'G', 'C', 'F', 'D', 'H', 'I',
                'D', 'A', 'I', 'H', 'G', 'C', 'B', 'F', 'B', 'C']

# N-back trial 划分
NBACK4_TRIALS = [7, 18, 9, 20]   # 4-back
NBACK1_TRIALS = [11, 2, 13]      # 1-back

N_back1 = []  # 存每个 1-back trial 的 fixation_duration_simple 列表
N_back4 = []  # 存每个 4-back trial 的 fixation_duration_simple 列表

# 新增：存每个 N-back 条件下每个 trial 的 saccade 平均速度
N_back1_saccade_vel = []
N_back4_saccade_vel = []


def remove_outliers(data):
    data = np.array(data)
    lower_bound = np.percentile(data, 5)
    upper_bound = np.percentile(data, 95)
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data


def find_c_segments1(sequence, time_sequence):  # 原函数先保留，用不到也无所谓
    segments = []
    start = None
    pd_flag = 0

    for i, element in enumerate(sequence):
        if 'C' in element and pd_flag == 0:
            if start is None:
                start = i
                pd_flag = 1

        elif 'C' not in element and pd_flag == 1:
            segments.append((start, i - 1))
            pd_flag = 0

    if start is not None:
        segments.append((start, len(sequence) - 1))

    return segments


# ====================== 主流程：遍历 1~30 号被试 ======================

path = os.getcwd()

# 如果需要回答数据可以加载，否则可注释掉
respondf = os.path.join(r"G:\myexperience\data\回答数据", "回答数据.xlsx")
respond_data = pd.read_excel(respondf)

need_four_square = [
    [9, 8, 4, 2], [7, 6, 3, 1], [8, 3, 9, 1], [2, 6, 4, 8],
    [8, 3, 2, 7], [3, 2, 1, 8], [2, 6, 8, 1], [4, 6, 1, 9],
    [2, 8, 4, 6], [9, 3, 7, 4]
]

for ID in range(1, 31):
    print("ID:", ID)

    # fixation（origin 版）
    fix_file = os.path.join(r'G:\myexperience', str(ID),
                            'fixation_calculate_angle_0.8_time_100ms_origin.xlsx')
    if not os.path.exists(fix_file):
        print("  缺少 fixation 文件:", fix_file)
        continue

    # saccade
    sacc_file = os.path.join(r'G:\myexperience', str(ID),
                             'saccade_calculate_angle_0.8_time_100ms.xlsx')
    if not os.path.exists(sacc_file):
        print("  缺少 saccade 文件:", sacc_file)
        continue

    fix_data = pd.read_excel(fix_file)
    sacc_data = pd.read_excel(sacc_file)

    # 遍历 20 个 trial（C1~C20 对应 groupi 0~19）
    for groupi in range(20):
        trial_index = groupi + 1

        # 只关心 N-back 相关 trial
        if (trial_index not in NBACK4_TRIALS) and (trial_index not in NBACK1_TRIALS):
            continue

        # fixation_duration_simple 列：字符串形式的 list
        fix_simple_raw = fix_data.loc[groupi, 'fixation_duration_simple']
        if isinstance(fix_simple_raw, str):
            fix_simple = ast.literal_eval(fix_simple_raw)
        elif isinstance(fix_simple_raw, (list, tuple)):
            fix_simple = fix_simple_raw
        else:
            fix_simple = []

        # 对应 saccade 平均速度
        sacc_vel_avg = sacc_data.loc[groupi, 'saccade_velocity_average']

        # N-back 分类
        if trial_index in NBACK4_TRIALS:
            N_back4.append(fix_simple)
            N_back4_saccade_vel.append(sacc_vel_avg)
        if trial_index in NBACK1_TRIALS:
            N_back1.append(fix_simple)
            N_back1_saccade_vel.append(sacc_vel_avg)


# ====================== 计算 N-back 指标 ======================

N_back1_fixation_number = []
N_back4_fixation_number = []
N_back1_fixation_average_duration = []
N_back4_fixation_average_duration = []
N_back1_fixation_max = []
N_back4_fixation_max = []

# 1-back fixation
for simple in N_back1:
    simple = np.array(simple, dtype=float)
    if simple.size == 0:
        N_back1_fixation_number.append(0)
        N_back1_fixation_average_duration.append(np.nan)
        N_back1_fixation_max.append(np.nan)
    else:
        N_back1_fixation_number.append(len(simple))
        N_back1_fixation_average_duration.append(simple.mean())
        N_back1_fixation_max.append(simple.max())

# 4-back fixation
for simple in N_back4:
    simple = np.array(simple, dtype=float)
    if simple.size == 0:
        N_back4_fixation_number.append(0)
        N_back4_fixation_average_duration.append(np.nan)
        N_back4_fixation_max.append(np.nan)
    else:
        N_back4_fixation_number.append(len(simple))
        N_back4_fixation_average_duration.append(simple.mean())
        N_back4_fixation_max.append(simple.max())

# saccade 平均速度
N_back1_saccade_average_velocity = N_back1_saccade_vel
N_back4_saccade_average_velocity = N_back4_saccade_vel

print("1-back trial 数:", len(N_back1))
print("4-back trial 数:", len(N_back4))
print("1-back saccade 平均速度前 5 个:", N_back1_saccade_average_velocity[:5])
print("4-back saccade 平均速度前 5 个:", N_back4_saccade_average_velocity[:5])
