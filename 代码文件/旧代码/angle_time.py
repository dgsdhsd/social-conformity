# -*- codeing = utf-8 -*-
# @Time : 2024/11/4 9:11
# @Author : 星空噩梦
# @File ： angle_time.py
# @Software : PyCharm

# -*- codeing = utf-8 -*-
# @Time : 2024/1/16 12:45
# @Author : 星空噩梦
# @File ： prefixation.py
# @Software : PyCharm

import os
import matplotlib
import pandas as pd
import math
import re
import numpy as np
#from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

agentrespond=['B','C','A','H','G','C','F','D','H','I','D','A','I','H','G','C','B','F','B','C']
list_true_easy=[[] for _ in range(9)]
list_true_hard=[[] for _ in range(9)]
list_flase_easy=[[] for _ in range(9)]
list_flase_hard=[[] for _ in range(9)]
trial_group=[11,2,13,7,18,9,20]

path = os.getcwd()
respondf = os.path.join(path, "../data", "回答数据", "回答数据.xlsx")
respond_data = pd.read_excel(respondf)


def calculate_angle(point1, point2, pivot_index=2):
    # 将输入点转换为 numpy 数组，方便操作
    point3=[-0.657, 1.109, -5.128]
    points = np.array([point1, point2, point3])

    # 设定支点和两个其它点的索引
    pivot = points[pivot_index]
    other_points = np.delete(points, pivot_index, axis=0)

    # 计算支点到其他两个点的向量
    vec1 = other_points[0] - pivot
    vec2 = other_points[1] - pivot

    # 计算两个向量的夹角
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 确保值在 [-1, 1] 之间

    # 转换为度数
    angle_degrees = np.degrees(angle)

    return angle_degrees

for Id in range(1,31):
    path=os.getcwd()
    directory_path = path+'/'+str(Id)  # 替换为你的目录路径
    file_list = os.listdir(directory_path)
    LIST=[]

    # 读取CSV文件
    fold='distance_1000_1000.xlsx'
    f = os.path.join(path, str(Id),fold)

    data = pd.read_excel(f)


    for idd in range(1,21):
        if idd not in trial_group:
            continue
        respond="回答"+str(idd)
        if 1 <= idd <= 5 or 11 <= idd <= 15:
            if respond_data[respond][Id-1] == agentrespond[idd-1]:
                for i in range(9):
                    list1=[data["C" + str(idd)+'x'][i + 1],data["C" + str(idd)+'y'][i + 1],data["C" + str(idd)+'z'][i + 1]]
                    list2 = [data["C" + str(idd) + 'x'][i], data["C" + str(idd) + 'y'][i],
                             data["C" + str(idd) + 'z'][i]]
                    list_true_easy[i].append(calculate_angle(list1,list2))
            elif respond_data[respond][Id-1] != agentrespond[idd-1]:
                for i in range(9):
                    list1 = [data["C" + str(idd) + 'x'][i + 1], data["C" + str(idd) + 'y'][i + 1],
                             data["C" + str(idd) + 'z'][i + 1]]
                    list2 = [data["C" + str(idd) + 'x'][i], data["C" + str(idd) + 'y'][i],
                             data["C" + str(idd) + 'z'][i]]
                    list_flase_easy[i].append(calculate_angle(list1, list2))
        elif 6 <= idd <= 10 or 16 <= idd <= 20:
            if respond_data[respond][Id-1] == agentrespond[idd-1]:
                for i in range(9):
                    list1 = [data["C" + str(idd) + 'x'][i + 1], data["C" + str(idd) + 'y'][i + 1],
                             data["C" + str(idd) + 'z'][i + 1]]
                    list2 = [data["C" + str(idd) + 'x'][i], data["C" + str(idd) + 'y'][i],
                             data["C" + str(idd) + 'z'][i]]
                    list_true_hard[i].append(calculate_angle(list1, list2))
            elif respond_data[respond][Id-1] != agentrespond[idd-1]:
                for i in range(9):
                    list1 = [data["C" + str(idd) + 'x'][i + 1], data["C" + str(idd) + 'y'][i + 1],
                             data["C" + str(idd) + 'z'][i + 1]]
                    list2 = [data["C" + str(idd) + 'x'][i], data["C" + str(idd) + 'y'][i],
                             data["C" + str(idd) + 'z'][i]]
                    list_flase_hard[i].append(calculate_angle(list1, list2))

df = pd.DataFrame({
    '1': list_true_easy[0],
    '2': list_true_easy[1],
    '3': list_true_easy[2],
    '4': list_true_easy[3],
    '5': list_true_easy[4],
    '6': list_true_easy[5],
    '7': list_true_easy[6],
    '8': list_true_easy[7],
    '9': list_true_easy[8]
    })

outf = os.path.join(path, "angle1.xlsx")
df.to_excel(outf, index=False)

df = pd.DataFrame({
    '1': list_flase_easy[0],
    '2': list_flase_easy[1],
    '3': list_flase_easy[2],
    '4': list_flase_easy[3],
    '5': list_flase_easy[4],
    '6': list_flase_easy[5],
    '7': list_flase_easy[6],
    '8': list_flase_easy[7],
    '9': list_flase_easy[8]
    })

outf = os.path.join(path, "angle2.xlsx")
df.to_excel(outf, index=False)

df = pd.DataFrame({
    '1': list_true_hard[0],
    '2': list_true_hard[1],
    '3': list_true_hard[2],
    '4': list_true_hard[3],
    '5': list_true_hard[4],
    '6': list_true_hard[5],
    '7': list_true_hard[6],
    '8': list_true_hard[7],
    '9': list_true_hard[8]
    })

outf = os.path.join(path, "angle3.xlsx")
df.to_excel(outf, index=False)

df = pd.DataFrame({
    '1': list_flase_hard[0],
    '2': list_flase_hard[1],
    '3': list_flase_hard[2],
    '4': list_flase_hard[3],
    '5': list_flase_hard[4],
    '6': list_flase_hard[5],
    '7': list_flase_hard[6],
    '8': list_flase_hard[7],
    '9': list_flase_hard[8]
    })

outf = os.path.join(path, "angle4.xlsx")
df.to_excel(outf, index=False)


