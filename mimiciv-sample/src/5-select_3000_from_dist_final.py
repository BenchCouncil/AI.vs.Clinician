import pandas as pd
import random
import os
from datetime import datetime, timedelta
from multiprocessing import Pool
import warnings
import csv
import ast
import numpy as np
import re

PATH = '/home/ddcui/'
ROOT = f'{PATH}mimic-original-data/mimiciv_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'


def distribution(df):
    if '年龄' in df.columns.tolist():
        age_list = list(df['年龄'])
    else:
        age_list = list(df['age'])

    age_1 = 0
    age_2_17 = 0
    age_18_30 = 0
    age_31_40 = 0
    age_41_50 = 0
    age_51_60 = 0
    age_61_70 = 0
    age_71_80 = 0
    age_81 = 0

    for age in age_list:
        if str(age) == '>89':
            age_81 = age_81 + 1
        else:
            age = int(float(age))
            if age <= 17:
                age_2_17 = age_2_17 + 1
            elif age <= 30 and age >= 18:
                age_18_30 = age_18_30 + 1
            elif age <= 40 and age >= 31:
                age_31_40 = age_31_40 + 1
            elif age <= 50 and age >= 41:
                age_41_50 = age_41_50 + 1
            elif age <= 60 and age >= 51:
                age_51_60 = age_51_60 + 1
            elif age <= 70 and age >= 61:
                age_61_70 = age_61_70 + 1
            elif age <= 80 and age >= 71:
                age_71_80 = age_71_80 + 1
            elif age >= 81:
                age_81 = age_81 + 1
    print(
        f'年龄<=17:  {age_2_17}, 年龄18-30:  {age_18_30}, 年龄31-40:  {age_31_40}, 年龄41-50:  {age_41_50}, 年龄51-60:  {age_51_60}, 年龄61-70:  {age_61_70}, 年龄71-80:  {age_71_80}, 年龄80岁以上:  {age_81}')

    if '性别' in df.columns.tolist():
        gender_list = list(df['性别'])
    else:
        gender_list = list(df['gender'])

    i = 0
    j = 0
    for gender in gender_list:
        if gender == '男' or gender == 'male':
            j = j + 1
        elif gender == '女' or gender == 'female':
            i = i + 1
    print(f'男:  {j},女:  {i}')

    if '体重' in df.columns.tolist():
        weight_list = list(df['体重'])
    else:
        weight_list = list(df['weight'])
    weight_0 = 0
    weight_1 = 0
    weight_30 = 0
    weight_31_40 = 0
    weight_41_50 = 0
    weight_51_60 = 0
    weight_61_70 = 0
    weight_71_80 = 0
    weight_81_90 = 0
    weight_91_100 = 0
    weight_101 = 0
    for weight in weight_list:
        if str(weight) == '体重未知':
            weight_0 = weight_0 + 1
        else:
            weight = float(weight)
            if weight <= 30:
                weight_30 = weight_30 + 1
            elif weight <= 40 and weight >= 31:
                weight_31_40 = weight_31_40 + 1
            elif weight <= 50 and weight >= 41:
                weight_41_50 = weight_41_50 + 1
            elif weight <= 60 and weight >= 51:
                weight_51_60 = weight_51_60 + 1
            elif weight <= 70 and weight >= 61:
                weight_61_70 = weight_61_70 + 1
            elif weight <= 80 and weight >= 71:
                weight_71_80 = weight_71_80 + 1
            elif weight <= 90 and weight >= 81:
                weight_81_90 = weight_81_90 + 1
            elif weight <= 100 and weight >= 91:
                weight_91_100 = weight_91_100 + 1
            elif weight >= 101:
                weight_101 = weight_101 + 1


    print(
        f'体重未知:  {weight_0}, 体重<=30kg:  {weight_30}, 体重31-40kg:  {weight_31_40}, 体重41-50kg:  {weight_41_50}, 体重51-60kg:  {weight_51_60}, 体重61-70kg:  {weight_61_70}, 体重71-80kg:  {weight_71_80}, 体重81-90kg:  {weight_81_90}, 体重91-100kg:  {weight_91_100}, 体重>100kg:  {weight_101}')

    # history_list = list(df['ILL_HISTORY'])
    # m = 0
    # n = 0
    # for history in history_list:
    #     if history == '有':
    #         n = n + 1
    #     elif history == '无':
    #         m = m + 1
    # print(f'有病史:  {n}, 无病史:  {m}')

    qsofa_list = list(df['qsofa_score'])
    score_0 = 0
    score_1 = 0
    score_2 = 0
    score_3 = 0
    for qsofa_score in qsofa_list:

        if qsofa_score == 'False':
            continue
        else:
            qsofa_score = int(qsofa_score)
        if qsofa_score == 0:
            score_0 = score_0 + 1
        elif qsofa_score == 1:
            score_1 = score_1 + 1
        elif qsofa_score == 2:
            score_2 = score_2 + 1
        elif qsofa_score == 3:
            score_3 = score_3 + 1
    print(f'qsofa分数 0:  {score_0}, qsofa分数 1:  {score_1}, qsofa分数 2:  {score_2}, qsofa分数 3:  {score_3}')

    df = df[~df['ill_time'].isna()]
    morta_list = list(df['morta_90'])
    dead_count = 0

    for i in morta_list:
        if i ==1:
            dead_count = dead_count+1
    ratio = (dead_count/len(morta_list))*100
    print(f'死亡率 {round(ratio,2)}%')

def get_3000_sample():
    #文件中存在的都是基本信息当前5 下一步检查当前4 数量
    df_ill_subjects = pd.read_csv(TO_ROOT+'preprocess/1-mimiciv_ill_redio_subject_data_distribute_havecxr.csv',encoding='gbk')
    df_not_subjects = pd.read_csv(TO_ROOT+'preprocess/1-mimiciv_not_redio_subject_data_distribute_havecxr.csv',encoding='gbk')

    df_ill_subjects['endtime'] = pd.to_datetime(df_ill_subjects['endtime'])
    df_ill_subjects = df_ill_subjects.drop_duplicates(subset=['start_endtime'])
    df_not_subjects['endtime'] = pd.to_datetime(df_not_subjects['endtime'])
    df_not_subjects = df_not_subjects.drop_duplicates(subset=['start_endtime'])

    df_check_54 = pd.read_csv(TO_ROOT+'preprocess/2-mimiciv_ill_and_not_have_cxr_5_4_check.csv',encoding='gbk')
    df_check_54['endtime'] = pd.to_datetime(df_check_54['endtime'])
    df_sepsis = df_ill_subjects[df_ill_subjects['endtime'].isin(set(df_check_54['endtime']))]
    def_not_sepsis = df_not_subjects[df_not_subjects['endtime'].isin(set(df_check_54['endtime']))]

    i = 1
    selected_subject_id = set()

    df_sample = pd.DataFrame()
    while i < 6:
        print(f'================第{i}组===============')
        # #200  0h
        df_sepsis_0h = df_sepsis[df_sepsis['time_range'] == '0h']
        df_sepsis_0h = df_sepsis_0h[~df_sepsis_0h['subject_id'].isin(selected_subject_id)]
        df_sepsis_0h = df_sepsis_0h.drop_duplicates(subset=['subject_id'])
        df_sepsis_0h = df_sepsis_0h.head(200)
        print(f'200  0h-----{len(df_sepsis_0h)}')
        selected_subject_id = set(df_sepsis_0h['subject_id']) | selected_subject_id

        #200中 20个 -3h 0h 3h 连续
        # 在此基础上 找3h 满足 5 4
        df_sepsis_0h_3h_before = df_sepsis[(df_sepsis['time_range'] == '3h') & (df_sepsis['hadm_id'].isin(set(df_sepsis_0h['hadm_id'])))]
        df_sepsis_0h_3h_before = df_sepsis_0h_3h_before.drop_duplicates(subset=['hadm_id'], keep='first')
        # 在此基础上  找-3h 满足 5 4
        df_sepsis_0h_3h_after = df_sepsis[(df_sepsis['time_range'] == '-3h') & (df_sepsis['hadm_id'].isin(set(df_sepsis_0h_3h_before['hadm_id'])))]
        df_sepsis_0h_3h_after = df_sepsis_0h_3h_after.drop_duplicates(subset=['hadm_id'], keep='last')

        connect_hadm_id = set(df_sepsis_0h_3h_before['hadm_id']) & set(df_sepsis_0h_3h_after['hadm_id'])
        df_sepsis_0h_3h_before = df_sepsis_0h_3h_before[df_sepsis_0h_3h_before['hadm_id'].isin(connect_hadm_id)]
        df_sepsis_0h_3h_after = df_sepsis_0h_3h_after[df_sepsis_0h_3h_after['hadm_id'].isin(connect_hadm_id)]
        df_sepsis_0h_3h_before = df_sepsis_0h_3h_before.drop_duplicates(subset=['hadm_id'])
        df_sepsis_0h_3h_after = df_sepsis_0h_3h_after.drop_duplicates(subset=['hadm_id'])

        df_sepsis_0h_3h_before = df_sepsis_0h_3h_before.head(20)
        df_sepsis_0h_3h_after = df_sepsis_0h_3h_after.head(20)

        print(f'df_sepsis_0h_3h_before----{len(df_sepsis_0h_3h_before)}')
        print(f'df_sepsis_0h_3h_after-----{len(df_sepsis_0h_3h_after)}')

        # 其他患病 30个-3h
        df_other_ill1 = df_sepsis[df_sepsis['time_range'] == '-3h']
        df_other_ill1 = df_other_ill1[~df_other_ill1['subject_id'].isin(selected_subject_id)]
        df_other_ill1 = df_other_ill1.drop_duplicates(subset=['subject_id'], keep='last')
        df_other_ill1 = df_other_ill1.head(30)
        selected_subject_id = set(df_other_ill1['subject_id']) | selected_subject_id
        print(f'df_other_ill -3h-----{len(df_other_ill1)}')

        #其他患病 30个3h
        df_other_ill2 = df_sepsis[df_sepsis['time_range'] == '3h']
        df_other_ill2 = df_other_ill2[~df_other_ill2['subject_id'].isin(selected_subject_id)]
        df_other_ill2 = df_other_ill2.drop_duplicates(subset=['subject_id'])
        df_other_ill2 = df_other_ill2.head(30)
        selected_subject_id = set(df_other_ill2['subject_id']) | selected_subject_id
        print(f'df_other_ill 3h----{len(df_other_ill2)}')

        df_ill_group = pd.concat([df_sepsis_0h,df_sepsis_0h_3h_before,df_sepsis_0h_3h_after,df_other_ill1,df_other_ill2])
        #300个正常
        df_not_group = def_not_sepsis[~def_not_sepsis['subject_id'].isin(selected_subject_id)]
        df_not_group = df_not_group.drop_duplicates(subset=['subject_id'],keep='last')
        df_not_group = df_not_group.sort_values('next_current_sum',ascending=False)
        df_not_group = df_not_group.head(300)
        selected_subject_id = set(df_not_group['subject_id']) | selected_subject_id
        print(f'df_not_sepsis_group----{len(df_not_group)}')

        df_ill_group['sepsis_label'] = 1
        df_not_group['sepsis_label'] = 0
        df_group = pd.concat([df_ill_group,df_not_group], ignore_index=True, sort=False)

        df_group['group'] = i
        i = i+1

        df_sample = pd.concat([df_sample, df_group], ignore_index=True, sort=False)
        distribution(df_group)

    df_a = df_sample.drop_duplicates(subset=['hadm_id'])
    print(f'选取的sample的hadm_id数量{len(df_a)}')
    print(f'选取的总sample数量{len(df_sample)}')

    df_sample.rename(columns={'TIME_RANGE': 'time_range'}, inplace=True)

    # df_sample.rename(columns={'性别': 'gender'}, inplace=True)
    # df_sample.rename(columns={'年龄': 'age'}, inplace=True)
    # df_sample.rename(columns={'体重': 'weight'}, inplace=True)
    # mapping_dict = {'男': 'male', '女': 'female'}
    # df_sample['gender'] = df_sample['gender'].map(mapping_dict)

    df_sample = df_sample.reindex(columns=['subject_id', 'hadm_id', '性别','年龄','体重','admittime', 'ill_time', 'morta_90','time_range' ,'start_endtime','starttime','endtime','sepsis_label', 'group'])
    del df_sample['QSOFA_SCORE']
    del df_sample['base_current_num']
    del df_sample['next_current_num']
    # 这个sample的选取会影响到后面的模型训练数据，如果重新测试的会带有随机性
    df_sample.to_csv(TO_ROOT+'preprocess/10-mimiciv_3000_sample.csv', encoding='gbk', mode='w', index=False)


if __name__ == '__main__':
    print('正在随机选取3000sample，满足相应人口学分布')
    get_3000_sample()
    print('end')