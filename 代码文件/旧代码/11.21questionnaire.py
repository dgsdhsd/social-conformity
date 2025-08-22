# -*- codeing = utf-8 -*-
# @Time : 2024/11/21 9:03
# @Author : 星空噩梦
# @File ： 11.21questionnaire.py
# @Software : PyCharm

'''
该文件为问卷以及t检验，采用配对t检验

'''

import pandas as pd
import numpy as np
import os

from scipy.stats import ttest_rel

true_answer=['B','A','A','H','G','C','B','D','B','I','B','A','A','H','G','C','B','D','B','I']

# Specify the file path
def calculate_stats(data1, data2):

    mean_std_1 = f"{np.mean(data1):.2f} ± {np.std(data1, ddof=1):.2f}"

    mean_std_2 = f"{np.mean(data2):.2f} ± {np.std(data2, ddof=1):.2f}"

    t_stat, p_value = ttest_rel(data1, data2)

    return mean_std_1, mean_std_2, p_value

file_path = os.path.join("G:\myexperience\data\问卷数据","试验后 序号.xlsx")
#file_path = r'/data/问卷数据/试验后 序号.xlsx'
# Attempt to open and read the Excel file
data = pd.read_excel(file_path)
back1_true = data["5、在1back实验中，我在实验给出的答案是正确的"].tolist()
back4_true = data["11、在4back实验中，我在实验给出的答案是正确的"].tolist()
back1_difficulty = data["6、在1back实验中，这场实验的难度"].tolist()
back4_difficulty = data["12、在4back实验中，这场实验的难度"].tolist()
back1_doubt = data["7、在1back实验中，我对我的回答的怀疑程度"].tolist()
back4_doubt = data["13、在4back实验中，我对我的回答的怀疑程度"].tolist()
back1_believe = data["8、在1back实验中，我对我的回答的自信程度"].tolist()
back4_believe = data["14、在4back实验中，我对我的回答的自信程度"].tolist()
back1_own = data["9、在1back实验中，我对这项实验给出的答案主要是基于我自己的观点。"].tolist()
back4_own = data["15、在4back实验中，我对这项实验给出的答案主要是基于我自己的观点。"].tolist()
back1_other = data["10、在1back实验中，实验中其他参与者的回答影响了我自己的回答。"].tolist()
back4_other = data["16、在4back实验中，实验中其他参与者的回答影响了我自己的回答。"].tolist()

#上面计算问卷信息，下面计算准确率

file_path = os.path.join("G:\myexperience\data\回答数据","回答数据.xlsx")
#file_path = r'/data/回答数据/回答数据.xlsx'

# Attempt to open and read the Excel file
data = pd.read_excel(file_path)
response_columns = [col for col in data.columns if '回答' in col]  # 筛选包含"回答"的列
response_lists = data[response_columns].apply(lambda row: row.tolist(), axis=1)

# 将结果转换为一个列表，其中每个元素是一个行的字母列表
back1_pro=[]
back4_pro=[]
result = response_lists.tolist()
for i in range(30):
    back1_num=0
    back4_num=0
    for j in range(0,5):
        if result[i][j]==true_answer[j]:
            back1_num+=1
    for j in range(10,15):
        if result[i][j]==true_answer[j%10]:
            back1_num+=1
    for j in range(5,10):
        if result[i][j]==true_answer[j%10]:
            back4_num+=1
    for j in range(15,20):
        if result[i][j]==true_answer[j%10]:
            back4_num+=1
    back1_pro.append(back1_num/10)
    back4_pro.append(back4_num / 10)

print("实验难度")
print(calculate_stats(back1_difficulty,back4_difficulty))
print("怀疑程度")
print(calculate_stats(back1_doubt,back4_doubt))
print("自信程度")
print(calculate_stats(back1_believe,back4_believe))
print("从众影响")
print(calculate_stats(back1_other,back4_other))
print("个人主导性")
print(calculate_stats(back4_own,back1_own))
print("主观准确率")
print(calculate_stats(back1_true,back4_true))
print("真实准确率")
print(calculate_stats(back1_pro,back4_pro))
print("一致性率")
#暂留

