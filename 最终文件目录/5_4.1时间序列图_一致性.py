# -*- codeing = utf-8 -*-
# @Time : 2025/4/1 8:11
# @Author : 星空噩梦
# @File ： 5_4.1时间序列图_一致性.py
# @Software : PyCharm


'''

修改smooth_data = gaussian_filter1d(prob_NONCONSISTENT[i], sigma=10)，260+行里的prob_NONCONSISTENT便可获得两张图。
'''

import os
import matplotlib
import pandas as pd
import numpy as np
#from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

#获取第i个数据
# 列出目录中的所有文件
Agentque=[2,3,1,8,7,3,6,4,8,9,4,1,9,8,7,3,2,6,2,3]
TrueQue=[2,1,1,8,7,3,2,4,2,9,2,1,1,8,7,3,2,4,2,9]

def resize_nearest_row(row, target_cols):
    """
    对单一行进行最近邻插值，调整列数到目标值。
    :param row: 单行数据（一维列表）
    :param target_cols: 目标列数
    :return: 调整后的行数据
    """
    current_cols = len(row)
    if current_cols == target_cols:
        return row.copy()
    indices = np.linspace(0, current_cols - 1, num=target_cols)
    indices = np.round(indices).astype(int)  # 四舍五入取最近的索引
    return [row[i] for i in indices]

def resize_2d_keep_rows(arr, target_cols):
    """
    保持行数不变，调整每一行的列数到目标值。
    :param arr: 原始二维列表（行数固定，列数可变）
    :param target_cols: 目标列数
    :return: 调整后的二维列表
    """
    return [resize_nearest_row(row, target_cols) for row in arr]

def convert_to_probability_matrix(consistent_resized):
    """
    将 58行×10000列 的二维列表转换为 5行×10000列 的概率矩阵。
    每列的概率表示该列中0~4的分布比例。
    """
    # 转换为NumPy数组
    arr = np.array(consistent_resized)
    n_rows, n_cols = arr.shape  # 原始形状：58行 × 10000列

    # 初始化结果矩阵：5行 × 10000列
    prob_matrix = np.zeros((5, n_cols), dtype=np.float64)

    # 遍历每一列，计算分类概率
    for col in range(n_cols):
        # 提取当前列的所有元素（共58个）
        column_data = arr[:, col]
        # 统计0~4的出现次数，minlength=5确保长度为5（即使某些值未出现）
        counts = np.bincount(column_data, minlength=5)
        # 计算概率（出现次数 / 总行数58）
        prob = counts / n_rows
        # 将概率存入结果矩阵的对应列
        prob_matrix[:, col] = prob

    # 转换为Python列表格式
    return prob_matrix.tolist()

CONSISTENT=[]
NONCONSISTENT=[]
number=0

trial_group=[7,18,9,20]
agentrespond=['B','C','A','H','G','C','F','D','H','I','D','A','I','H','G','C','B','F','B','C']

def find_c_segments(sequence):
    """
    Finds continuous segments in a sequence where all elements contain 'C'.
    Returns a list of tuples (start_index, end_index) for each segment.

    :param sequence: List of strings
    :return: List of tuples with start and end indices
    """
    segments = []
    start = None  # Track the start of a segment

    for i, element in enumerate(sequence):
        if 'C' in element:
            if start is None:  # Start of a new segment
                start = i
        else:
            if start is not None:  # End of a segment
                segments.append((start, i - 1))
                start = None

    # Add the last segment if the sequence ends with a 'C' segment
    if start is not None:
        segments.append((start, len(sequence) - 1))

    return segments
path=os.getcwd()
need_four_square=[[9, 8, 4, 2],[7, 6, 3, 1],[8, 3, 9, 1],[2, 6, 4, 8],[8, 3, 2, 7],[3, 2, 1, 8],[2, 6, 8, 1],[4, 6, 1, 9],[2, 8, 4, 6],[9, 3, 7, 4]]

#respondf = os.path.join("/data/回答数据", "回答数据.xlsx")
respondf = os.path.join("G:\myexperience\data\回答数据","回答数据.xlsx")

respond_data = pd.read_excel(respondf)


for ID in range(1,31):
    # 读取CSV文件
    print(ID)
    LSATLIST=[]
    f = os.path.join("G:\myexperience", str(ID), 'fixation_angle_0.8_time_100ms.xlsx')
    data = pd.read_excel(f)
    for groupi in range(20):
        lists = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        AOI_time=[]
        state_list = data['state'].tolist()
        bzstate=find_c_segments(state_list)
        for i in range(bzstate[groupi][0], bzstate[groupi][1] + 1):
            time = (data['time'][i] - data['time'][i - 1])
            four_squre = need_four_square[groupi % 10]
            if data.at[i, 'gaze_x'] >= 3.2:
                if 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[8] += time
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==8:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)
                elif 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[7] += time
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==7:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)
                elif 0.525 <= data.at[i, 'gaze_y'] <= 0.905 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[6] += time
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==6:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)

                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[5] += time
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==5:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[4] += time
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==4:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)
                elif 0.905 <= data.at[i, 'gaze_y'] <= 1.285 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[3] += time
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==3:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)

                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -7.345 <= data.at[i, 'gaze_z'] <= -6.965:
                    lists[2] += time
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==2:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -6.965 <= data.at[i, 'gaze_z'] <= -6.585:
                    lists[1] += 1
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==1:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)
                elif 1.285 <= data.at[i, 'gaze_y'] <= 1.665 and -6.585 <= data.at[i, 'gaze_z'] <= -6.205:
                    lists[0] += time
                    pdd=0
                    for group_i in range(4):
                        if four_squre[group_i] - 1==0:
                            AOI_time.extend([group_i] * time)
                            pdd=1
                            break
                    if pdd==0:
                        AOI_time.extend([4] * time)
                else:
                    lists[9] += time
                    AOI_time.extend([4] * time)

            else:
                lists[9] += time
                AOI_time.extend([4] * time)
                continue
        if groupi+1 in trial_group:
            if respond_data.iloc[ID-1, 2+groupi] == agentrespond[groupi]:
                CONSISTENT.append(AOI_time)
                print(number)
                number+=1
            else :
                NONCONSISTENT.append(AOI_time)
                print(number)
                number+=1



CONSISTENT_RESIZED = resize_2d_keep_rows(CONSISTENT, target_cols=10000)
NONCONSISTENT_RESIZED = resize_2d_keep_rows(NONCONSISTENT, target_cols=10000)


prob_CONSISTENT = convert_to_probability_matrix(CONSISTENT_RESIZED)
prob_NONCONSISTENT = convert_to_probability_matrix(NONCONSISTENT_RESIZED)




time = np.arange(10000)

# 定义标签
labels = ['4back',
          '3back',
          '2back',
          '1back',
          'Other']

plt.figure(figsize=(8, 6))
for i in range(5):
    # 使用高斯平滑
    smooth_data = gaussian_filter1d(prob_NONCONSISTENT[i], sigma=10)
    plt.plot(time, smooth_data, label=labels[i], marker='o' if i == 0 else None)

# # 标注平均 target offset
# plt.axvline(x=200, color='gray', linestyle='--')
# plt.text(210, 0.7, 'Average target offset', color='gray', fontsize=10)

# 设置轴标签和标题
plt.xlabel('Time since target onset (ms)')
plt.ylabel('Fixation probability')

# 显示图例
plt.legend()

# 显示图形
plt.show()
