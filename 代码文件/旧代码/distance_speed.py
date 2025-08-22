# -*- codeing = utf-8 -*-
# @Time : 2023/12/8 9:23
# @Author : 星空噩梦
# @File ： generator.py
# @Software : PyCharm

import os
import matplotlib
import pandas as pd
import numpy as np
#from sklearn.cluster import KMeans
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#获取第i个数据
# 列出目录中的所有文件
path=os.getcwd()


# 查找匹配的文件
# if(TASKi<10):
#     matching_files = [filename for filename in file_list if filename.startswith(f"i0{TASKi}_")]
# if (TASKi >= 10):
#         matching_files = [filename for filename in file_list if filename.startswith(f"i{TASKi}_")]
# print(matching_files)

# 读取CSV文件
NAME=['C_11','C_21','C_31','C_41','C_51','C_61','C_71','C_81','C_91','C_101','C_111','C_121','C_131','C_141','C_151','C_161','C_171','C_181','C_191','C_201']




def biao(time):
    Time=abs(time - data['time'][0])
    current=0
    for step in range(1,len(data)):
        if Time>=abs(time-data['time'][step]):
            current=step
            Time=abs(time-data['time'][step])
    return current

def pupil(first,last):
    pup1=0   #左
    pup2 = 0 #右
    pup3=0   #平均
    for i in range(first,last+1):
        pup1+=data['gaze_x'][i]
        pup2+=data['gaze_y'][i]
        pup3+=data['gaze_z'][i]

    pup1/=(last+1-first)
    pup2/=(last+1-first)
    pup3/=(last+1-first)

    return [pup1,pup2,pup3]


for id in range(1,31):
    ID=str(id)
    fold='fixation_angle_0.8_time_100ms.xlsx'

    f = os.path.join(path, ID, fold)
    data = pd.read_excel(f)

    gaze_x=[]
    gaze_y=[]
    gaze_z=[]

    for name in NAME:
        list1 = []  # 左
        list2 = []  # 右
        list3 = []  # 平均
        first=0
        time=0
        for step in range(len(data)):
            if data['state'][step]==name:
                first=step
                break
            if step==len(data)-1:
                print(name)
        time=data['time'][first]
        list=pupil(first,biao(time+1000))
        list1.append(list[0])
        list2.append(list[1])
        list3.append(list[2])
        for j in range(9):
            first=biao(time+1000)
            time=data['time'][first]
            list = pupil(first, biao(time + 1000))
            list1.append(list[0])
            list2.append(list[1])
            list3.append(list[2])

        gaze_x.append(list1)
        gaze_y.append(list2)
        gaze_z.append(list3)

    ff = os.path.join(path, ID)
    if not os.path.exists(ff):
        os.mkdir(ff)
    dfleft = pd.DataFrame({
        'C1x': gaze_x[0],
        'C1y': gaze_y[0],
        'C1z': gaze_z[0],
        'C2x': gaze_x[1],
        'C2y': gaze_y[1],
        'C2z': gaze_z[1],
        'C3x': gaze_x[2],
        'C3y': gaze_y[2],
        'C3z': gaze_z[2],
        'C4x': gaze_x[3],
        'C4y': gaze_y[3],
        'C4z': gaze_z[3],
        'C5x': gaze_x[4],
        'C5y': gaze_y[4],
        'C5z': gaze_z[4],
        'C6x': gaze_x[5],
        'C6y': gaze_y[5],
        'C6z': gaze_z[5],
        'C7x': gaze_x[6],
        'C7y': gaze_y[6],
        'C7z': gaze_z[6],
        'C8x': gaze_x[7],
        'C8y': gaze_y[7],
        'C8z': gaze_z[7],
        'C9x': gaze_x[8],
        'C9y': gaze_y[8],
        'C9z': gaze_z[8],
        'C10x': gaze_x[9],
        'C10y': gaze_y[9],
        'C10z': gaze_z[9],
        'C11x': gaze_x[10],
        'C11y': gaze_y[10],
        'C11z': gaze_z[10],
        'C12x': gaze_x[11],
        'C12y': gaze_y[11],
        'C12z': gaze_z[11],
        'C13x': gaze_x[12],
        'C13y': gaze_y[12],
        'C13z': gaze_z[12],
        'C14x': gaze_x[13],
        'C14y': gaze_y[13],
        'C14z': gaze_z[13],
        'C15x': gaze_x[14],
        'C15y': gaze_y[14],
        'C15z': gaze_z[14],
        'C16x': gaze_x[15],
        'C16y': gaze_y[15],
        'C16z': gaze_z[15],
        'C17x': gaze_x[16],
        'C17y': gaze_y[16],
        'C17z': gaze_z[16],
        'C18x': gaze_x[17],
        'C18y': gaze_y[17],
        'C18z': gaze_z[17],
        'C19x': gaze_x[18],
        'C19y': gaze_y[18],
        'C19z': gaze_z[18],
        'C20x': gaze_x[19],
        'C20y': gaze_y[19],
        'C20z': gaze_z[19],

    })
    ffleft=os.path.join(ff,'distance_1000_1000.xlsx')
    dfleft.to_excel(ffleft, index=False)
    print('LKL')











