# -*- codeing = utf-8 -*-
# @Time : 2025/8/11 9:36
# @Author : 星空噩梦
# @File ： 0_问卷信息.py
# @Software : PyCharm

"""
该文件用于分析问卷数据并进行配对 t 检验
可直接运行，查看控制台输出结果。
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# 正确答案序列
TRUE_ANSWERS = ['B', 'A', 'A', 'H', 'G', 'C', 'B', 'D', 'B', 'I',
                'B', 'A', 'A', 'H', 'G', 'C', 'B', 'D', 'B', 'I']

Consisten_Answers= ['B', 'C', 'A', 'H', 'G', 'C', 'F', 'D', 'H', 'I', 'D', 'A', 'I', 'H', 'G', 'C', 'B', 'F', 'B', 'C']


def calculate_stats(data1, data2):
    """计算两组数据的均值±标准差以及配对 t 检验 p 值"""
    mean_std_1 = f"{np.mean(data1):.2f} ± {np.std(data1, ddof=1):.2f}"
    mean_std_2 = f"{np.mean(data2):.2f} ± {np.std(data2, ddof=1):.2f}"
    _, p_value = ttest_rel(data1, data2)
    return mean_std_1, mean_std_2, p_value


# ======== 读取问卷数据 ========
questionnaire_path = os.path.join(
    r"G:\myexperience\data\问卷数据",
    "试验后 序号.xlsx"
)
questionnaire_df = pd.read_excel(questionnaire_path)

back1_true = questionnaire_df["5、在1back实验中，我在实验给出的答案是正确的"].tolist()
back4_true = questionnaire_df["11、在4back实验中，我在实验给出的答案是正确的"].tolist()
back1_difficulty = questionnaire_df["6、在1back实验中，这场实验的难度"].tolist()
back4_difficulty = questionnaire_df["12、在4back实验中，这场实验的难度"].tolist()
back1_doubt = questionnaire_df["7、在1back实验中，我对我的回答的怀疑程度"].tolist()
back4_doubt = questionnaire_df["13、在4back实验中，我对我的回答的怀疑程度"].tolist()
back1_confidence = questionnaire_df["8、在1back实验中，我对我的回答的自信程度"].tolist()
back4_confidence = questionnaire_df["14、在4back实验中，我对我的回答的自信程度"].tolist()
back1_self_opinion = questionnaire_df["9、在1back实验中，我对这项实验给出的答案主要是基于我自己的观点。"].tolist()
back4_self_opinion = questionnaire_df["15、在4back实验中，我对这项实验给出的答案主要是基于我自己的观点。"].tolist()
back1_social_influence = questionnaire_df["10、在1back实验中，实验中其他参与者的回答影响了我自己的回答。"].tolist()
back4_social_influence = questionnaire_df["16、在4back实验中，实验中其他参与者的回答影响了我自己的回答。"].tolist()


# ======== 计算真实准确率 ========
response_path = os.path.join(
    r"G:\myexperience\data\回答数据",
    "回答数据.xlsx"
)
response_df = pd.read_excel(response_path)

# 选出包含“回答”的列
response_columns = [col for col in response_df.columns if '回答' in col]
response_lists = response_df[response_columns].apply(lambda row: row.tolist(), axis=1).tolist()

back1_accuracy = []
back4_accuracy = []

for i in range(30):
    back1_correct_count = 0
    back4_correct_count = 0

    # 1back 前5列 & 10~15列
    for j in range(0, 5):
        if response_lists[i][j] == TRUE_ANSWERS[j]:
            back1_correct_count += 1
    for j in range(10, 15):
        if response_lists[i][j] == TRUE_ANSWERS[j % 10]:
            back1_correct_count += 1

    # 4back 中间5列 & 15~20列
    for j in range(5, 10):
        if response_lists[i][j] == TRUE_ANSWERS[j % 10]:
            back4_correct_count += 1
    for j in range(15, 20):
        if response_lists[i][j] == TRUE_ANSWERS[j % 10]:
            back4_correct_count += 1

    back1_accuracy.append(back1_correct_count / 10)
    back4_accuracy.append(back4_correct_count / 10)

back1_consistent = []
back4_consistent = []
trial_group_back4=[7,18,9,20]
trial_group_back1=[2,11,13]

for i in range(30):
    back1_consistent_count = 0
    back4_consistent_count = 0

    for j in trial_group_back1:
        if response_lists[i][j-1] == Consisten_Answers[j-1]:
            back1_consistent_count+=1
    for j in trial_group_back4:
        if response_lists[i][j-1] == Consisten_Answers[j-1]:
            back4_consistent_count+=1

    back1_consistent.append(back1_consistent_count / 3)
    back4_consistent.append(back4_consistent_count / 4)


# ======== 输出结果 ========
print("实验难度", calculate_stats(back1_difficulty, back4_difficulty))
print("怀疑程度", calculate_stats(back1_doubt, back4_doubt))
print("自信程度", calculate_stats(back1_confidence, back4_confidence))
print("从众影响", calculate_stats(back1_social_influence, back4_social_influence))
print("个人主导性", calculate_stats(back4_self_opinion, back1_self_opinion))
print("主观准确率", calculate_stats(back1_true, back4_true))
print("真实准确率", calculate_stats(back1_accuracy, back4_accuracy))
print("一致性率",calculate_stats(back1_consistent, back4_consistent))


