# -*- codeing = utf-8 -*-
# @Time : 2025/4/1 8:49
# @Author : 星空噩梦
# @File ： 4.1一致性_SGE.py
# @Software : PyCharm


import os
import matplotlib
import pandas as pd
import numpy as np
#from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro

#获取第i个数据
# 列出目录中的所有文件
Agentque=[2,3,1,8,7,3,6,4,8,9,4,1,9,8,7,3,2,6,2,3]
TrueQue=[2,1,1,8,7,3,2,4,2,9,2,1,1,8,7,3,2,4,2,9]
agentrespond=['B','C','A','H','G','C','F','D','H','I','D','A','I','H','G','C','B','F','B','C']
CONSISTENT=[]
NONCONSISTENT=[]
trial_group=[7,18,9,20]
def remove_outliers(data):
    data = np.array(data)  # 确保数据是numpy数组
    lower_bound = np.percentile(data, 5)
    upper_bound = np.percentile(data, 95)
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data
def calculate_entropy(values):
    # 过滤掉 0 值，避免 log2(0) 的问题
    values = [v for v in values if v != 0]

    # 检查是否所有值都为 0
    if not values:
        print("出现问题时 values:", values)  # 输出调试信息
        return 0.0  # 如果所有值都是 0，返回熵为 0

    # 计算总和
    total = sum(values)

    # 检查总和是否为 0
    if total == 0:
        print("出现问题时 values:", values)  # 输出调试信息
        return 0.0  # 如果总和为 0，返回熵为 0

    # 计算概率
    probabilities = [v / total for v in values]

    # 计算信息熵
    entropy = -sum(p * np.log2(p) for p in probabilities if p != 0)  # 再次确保 p 不为 0

    # 检查熵是否为 NaN
    if np.isnan(entropy):
        print("出现问题时 values:", values)  # 输出调试信息
        print("计算得到的概率:", probabilities)  # 输出概率以进一步排查问题

    return entropy

def generalized_entropy(counts, alpha):
    """
    Calculate the generalized entropy (Rényi entropy) using log base 2.

    Parameters:
    counts (list or np.ndarray): List of class counts (non-negative numbers).
    alpha (float): Entropy order parameter.
                   - For alpha = 1 → Shannon entropy (log2-based)
                   - For alpha > 0 and alpha ≠ 1 → Rényi entropy

    Returns:
    float: Generalized entropy value (in bits)
    """
    counts = np.array(counts, dtype=np.float64)

    # 处理负数和全零情况
    if np.any(counts < 0):
        raise ValueError("Counts must be non-negative")
    total = np.sum(counts)
    if total == 0:
        raise ValueError("Total count cannot be zero")

    # 计算概率分布
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]  # 去除0，避免 log(0)

    if alpha <= 0:
        raise ValueError("Alpha must be greater than 0")

    if alpha == 1:
        # Shannon 熵（log2）
        entropy = -np.sum(probabilities * np.log2(probabilities))
    else:
        # Rényi 熵（log2）
        entropy = (1 / (1 - alpha)) * np.log2(np.sum(probabilities ** alpha))

    return entropy


def find_c_segments1(sequence,time_sequence):#这是整体的
    """
    Finds continuous segments in a sequence where all elements contain 'C'.
    Returns a list of tuples (start_index, end_index) for each segment.

    :param sequence: List of strings
    :return: List of tuples with start and end indices
    """
    segments = []
    start = None  # Track the start of a segment
    time = None
    pd=0

    for i, element in enumerate(sequence):
        if 'C' in element and pd==0:
            if start is None:  # Start of a new segment
                start = i
                time=time_sequence[i]+6000
                pd=1
        elif 'C' in element and pd==1:
            if time_sequence[i]>time:
                segments.append((start, i - 1))
                start = None
                pd=2

        elif 'C' not in element and pd==2:
            pd=0

    # Add the last segment if the sequence ends with a 'C' segment
    if start is not None:
        segments.append((start, len(sequence) - 1))

    return segments
def find_c_segments2(sequence,time_sequence):
    """
    Finds continuous segments in a sequence where all elements contain 'C'.
    Returns a list of tuples (start_index, end_index) for each segment.

    :param sequence: List of strings
    :return: List of tuples with start and end indices
    """
    segments = []
    start = None  # Track the start of a segment
    true_start=None
    time_first = None
    time_last = None
    pd=0

    for i, element in enumerate(sequence):
        if 'C' in element and pd==0:
            if start is None:  # Start of a new segment
                start = i
                time_first=time_sequence[i]+4000
                time_last=time_first+3000
            elif true_start is None and time_sequence[i]>time_first:
                true_start=i

            elif time_sequence[i]>time_last:
                segments.append((true_start, i - 1))
                if time_sequence[i - 1]-time_sequence[true_start]>5000:
                    print(start,true_start,time_first,time_last,pd)
                start = None  # Track the start of a segment
                true_start = None
                time_first = None
                time_last = None
                pd=1
        else :
            pd = 0
            start = None  # Track the start of a segment
            true_start = None
            time_first = None
            time_last = None


    # Add the last segment if the sequence ends with a 'C' segment
    if start is not None:
        segments.append((start, len(sequence) - 1))

    return segments

def find_c_segments3(sequence,time_sequence):
    """
    Finds continuous segments in a sequence where all elements contain 'C'.
    Returns a list of tuples (start_index, end_index) for each segment.

    :param sequence: List of strings
    :return: List of tuples with start and end indices
    """
    segments = []
    start = None  # Track the start of a segment
    time=0
    true_start=None

    for i, element in enumerate(sequence):
        if 'C' in element:
            if start is None:  # Start of a new segment
                start = i
                time=time_sequence[i]+7000
            elif time_sequence[i]>time and true_start is None:
                true_start=i
        else:
            if start is not None:  # End of a segment
                segments.append((true_start, i - 1))
                start = None
                true_start=None

    # Add the last segment if the sequence ends with a 'C' segment
    if true_start is not None:
        segments.append((true_start, len(sequence) - 1))

    return segments
path=os.getcwd()
respondf = os.path.join("G:\myexperience\data\回答数据","回答数据.xlsx")
#respondf = os.path.join("/data/回答数据", "回答数据.xlsx")

respond_data = pd.read_excel(respondf)
need_four_square=[[9, 8, 4, 2],[7, 6, 3, 1],[8, 3, 9, 1],[2, 6, 4, 8],[8, 3, 2, 7],[3, 2, 1, 8],[2, 6, 8, 1],[4, 6, 1, 9],[2, 8, 4, 6],[9, 3, 7, 4]]
for ID in range(1,31):
    # 读取CSV文件
    print(ID)
    LSATLIST=[]
    f = os.path.join('G:\myexperience', str(ID), 'fixation_angle_0.8_time_100ms.xlsx')
    data = pd.read_excel(f)
    for groupi in range(20):
        lists = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        state_list = data['state'].tolist()
        time_list=data['time'].tolist()
        bzstate=find_c_segments1(state_list,time_list)#这是前面的
        #bzstate = find_c_segments2(state_list,time_list)#中间趋势
        #bzstate = find_c_segments3(state_list, time_list)#最后一个进行特别的函数
        for i in range(bzstate[groupi][0], bzstate[groupi][1] + 1):
            time = (data['time'][i] - data['time'][i - 1])
            if data.at[i, 'gaze_x'] >= 3.2:
                if 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[8] += time
                elif 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[7] += time
                elif 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[6] += time

                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[5] += time
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[4] += time
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[3] += time

                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[2] += time
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[1] += time
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[0] += time
                else:
                    lists[9] += time

            else:
                lists[9] += time
                continue
        four_squre = need_four_square[groupi%10]
        lists[10] = lists[four_squre[0] - 1]
        lists[11] = lists[four_squre[1] - 1]
        lists[12] = lists[four_squre[2] - 1]
        lists[13] = lists[four_squre[3] - 1]
        lists.append(lists[0]+lists[1]+lists[2]+lists[3]+lists[4]+lists[5]+lists[6]+lists[7]+lists[8]+lists[9]-lists[10]-lists[11]-lists[12]-lists[13])
        LSATLIST.append(lists)
        if groupi+1 in trial_group:
            if respond_data.iloc[ID - 1, 2 + groupi] == agentrespond[groupi]:
                CONSISTENT.append(lists)
            else:
                NONCONSISTENT.append(lists)
        # #上面是一致性专属
        # if groupi+1 in [11,2,13]:
        #     CONSISTENT.append(lists)   #CONSISTENT是1back，NONCONSISTENT是4back
        # if groupi+1 in [7,18,9,20]:
        #     NONCONSISTENT.append(lists)


# entropies_CONSISTENT = []
# for row in CONSISTENT:
#     last_five = row[-5:]  # 提取最后 5 个数值
#     entropy = calculate_entropy(last_five)  # 计算信息熵
#     entropies_CONSISTENT.append(entropy)
#
# entropies_NONCONSISTENT = []
# for row in NONCONSISTENT:
#     last_five = row[-5:]  # 提取最后 5 个数值
#     entropy = calculate_entropy(last_five)  # 计算信息熵
#     entropies_NONCONSISTENT.append(entropy)

alphas = [0.5, 0.7, 1, 2, 3]

for alpha in alphas:
    entropies_CONSISTENT = []
    entropies_NONCONSISTENT = []

    for row in CONSISTENT:
        last_five = row[-5:]
        entropy = generalized_entropy(last_five, alpha)
        entropies_CONSISTENT.append(entropy)

    for row in NONCONSISTENT:
        last_five = row[-5:]
        entropy = generalized_entropy(last_five, alpha)
        entropies_NONCONSISTENT.append(entropy)

    # 计算统计指标
    mean_consistent = np.mean(entropies_CONSISTENT)
    mean_nonconsistent = np.mean(entropies_NONCONSISTENT)

    std_consistent = np.std(entropies_CONSISTENT)
    std_nonconsistent = np.std(entropies_NONCONSISTENT)

    offset = mean_consistent - mean_nonconsistent

    t_statistic, p_value_t = ttest_ind(entropies_CONSISTENT, entropies_NONCONSISTENT)

    # 输出结果
    print(f"\n=== α = {alpha} ===")
    print("CONSISTENT 组均值:", mean_consistent)
    print("NONCONSISTENT 组均值:", mean_nonconsistent)
    print("CONSISTENT 组标准差:", std_consistent)
    print("NONCONSISTENT 组标准差:", std_nonconsistent)
    print("偏移值 (CONSISTENT - NONCONSISTENT):", offset)
    print("T 检验结果:")
    print("t 统计量:", t_statistic)
    print("p 值:", p_value_t)








