import pickle
import math
import re
import csv
import concurrent.futures
import os
from functools import reduce
import multiprocessing

from operator import add
import pandas as pd
import numpy as np
import random
import copy
import sys

PATH = '/home/ddcui/'
ROOT = f'{PATH}/hai-med-database/mimic-lstm/data/mimiciii_preprocess/'

def process_model_input(df_ill_subjects_chart,filename):

    print('==============================================')
    print(f'开始根据HADMID_DAY 进行sample 填充，每个sample的步长为336')
    # df_ill_subjects_chart['HADMID_SAMPLE'] = df_ill_subjects_chart['HADMID_DAY'].apply(lambda x: x.split('#')[0])
    grouped_df = df_ill_subjects_chart.groupby(['hadmid_sample'])
    print(f'----填充前 HADMID_DAY sample总数 {len(grouped_df)}')
    with multiprocessing.Pool() as pool:
        groups = pool.map(fill_group_hadmid_day,
                              [grouped_df.get_group(group_key) for group_key in grouped_df.groups])
    fill_df_samples = pd.concat(groups)
    fill_df_samples = remove_data(fill_df_samples)
    grouped_df = fill_df_samples.groupby(['hadmid_sample'])
    print(f'----填充后 HADMID_DAY sample总数 {len(grouped_df)-1}')
    del fill_df_samples['hadmid_sample']
    # 统一调整列的数据
    columns_str = '''chronic airway obstruction,chronic bronchitis,chronic pancreatitis,chronic kidney disease,chronic liver disease,chronic pulmonary,diabetes,emphysema,end stage renal disease,heart failure,immunodeficiency,organ insufficiency,rena insufficiency,biliary complaint,cardiac complaint,dementia complaint,fall complaint,gastrointestinal bleed complaint,seizure complaint,stroke complain,gender,weight,alanine transaminase,amylase,arterial pH,aspartate aminotransferase,bicarbonate,bun,bun/creatinine ratio,dbp,fio2,gcs,heart rate,hematocrit,hemoglobin,lactate,lipase,oxygen saturation,arterial PaCO2,pao2/fio2 ratio,map,platelet count,potassium,rass,respiratory rate,shock index,sodium,sbp,temperature,wbc,ill_label'''
    columns_list = columns_str.split(',')
    fill_df_samples = fill_df_samples[columns_list]
    fill_df_samples.to_csv(filename, mode='w',index=False)

def fill_group_hadmid_day(group_ill):
    if group_ill.shape[0] == 0:
        return pd.DataFrame(columns=group_ill.colimns)
    if group_ill.shape[0] < 336:
        num_rows_to_add = 336 - group_ill.shape[0]
        rows_to_add = pd.DataFrame([[-4] * group_ill.shape[1]] * num_rows_to_add, columns=group_ill.columns)
        group_ill = pd.concat([group_ill, rows_to_add], ignore_index=True)
    elif group_ill.shape[0] > 336:
        group_ill = group_ill.head(336)
    return group_ill

def remove_data(df):
    columns_list = list(df.columns)
    if 'subject_id' in columns_list:
        del df['subject_id']
    if 'hadm_id' in columns_list:
        del df['hadm_id']
    if 'admittime' in columns_list:
        del df['admittime']
    if 'illtime' in columns_list:
        del df['illtime']
    if 'starttime' in columns_list:
        del df['starttime']
    if 'endtime' in columns_list:
        del df['endtime']
    if 'period' in columns_list:
        del df['period']

    return df

#执行脚本的时候获取
# arguments = sys.argv
# miss_rate = float(arguments[1])
miss_rate = 0.4
kind = 'diff_meanvalue'
df_ill = pd.read_csv(ROOT+f'lstm_ill_model_input_data_by_{kind}.csv')
df_not = pd.read_csv(ROOT+f'lstm_not_model_input_data_by_{kind}.csv')

#现在的train val test的id已经是把缺失率大于40的都删掉了

#弄 3h  0h  -3h样本
def time_period_3h_label():
    print('正在处理患病患者的3h ')
    df_3h_endtime = df_ill[df_ill['period'] == '3h']
    df_3h_endtime = df_3h_endtime.sample(frac=1, random_state=42)
    df_3h_endtime = df_3h_endtime.drop_duplicates(subset=['hadm_id'],keep='first')
    df_3h_endtime['3h_endtime'] = df_3h_endtime['endtime']
    df_3h = pd.merge(df_ill,df_3h_endtime[['hadm_id','3h_endtime']],on=['hadm_id'],how='left')
    df_3h['3h_endtime'] = pd.to_datetime(df_3h['3h_endtime'])
    df_3h['endtime'] = pd.to_datetime(df_3h['endtime'])
    df_3h = df_3h[df_3h['3h_endtime']>=df_3h['endtime']]
    df_3h['ill_label'] = 1
    df_3h = df_3h.sort_values(by=['hadm_id', 'starttime'] ,ascending=[True, True])
    del df_3h['3h_endtime']
    df_3h['hadmid_sample'] = df_3h['hadm_id'].astype(str) + '3h'
    return df_3h

def time_period_0h_label():
    print('正在处理患病患者的0h ')
    df_0h_endtime = df_ill[df_ill['period'] == '0h']
    df_0h_endtime = df_0h_endtime.drop_duplicates(subset=['hadm_id'], keep='first')
    df_0h_endtime['0h_endtime'] = df_0h_endtime['endtime']
    df_0h = pd.merge(df_ill, df_0h_endtime[['hadm_id', '0h_endtime']], on=['hadm_id'], how='left')
    df_0h['0h_endtime'] = pd.to_datetime(df_0h['0h_endtime'])
    df_0h['endtime'] = pd.to_datetime(df_0h['endtime'])
    df_0h = df_0h[df_0h['0h_endtime'] >= df_0h['endtime']]
    df_0h['ill_label'] = 1
    df_0h = df_0h.sort_values(by=['hadm_id', 'starttime'],ascending=[True, True])
    del df_0h['0h_endtime']
    df_0h['hadmid_sample'] = df_0h['hadm_id'].astype(str) + '0h'

    return df_0h

def time_period_3h_back_label():
    print('正在处理患病患者的-3h ')

    df_3h_back_endtime = df_ill[df_ill['period'] == '-3h']
    df_3h_back_endtime = df_3h_back_endtime.sample(frac=1, random_state=42)
    df_3h_back_endtime = df_3h_back_endtime.drop_duplicates(subset=['hadm_id'], keep='first')
    df_3h_back_endtime['3h_endtime'] = df_3h_back_endtime['endtime']
    df_3h_back = pd.merge(df_ill, df_3h_back_endtime[['hadm_id', '3h_endtime']], on=['hadm_id'], how='left')
    df_3h_back['3h_endtime'] = pd.to_datetime(df_3h_back['3h_endtime'])
    df_3h_back['endtime'] = pd.to_datetime(df_3h_back['endtime'])
    df_3h_back = df_3h_back[df_3h_back['3h_endtime'] >= df_3h_back['endtime']]
    df_3h_back['ill_label'] = 1
    df_3h_back = df_3h_back.sort_values(by=['hadm_id', 'starttime'],ascending=[True, True])
    del df_3h_back['3h_endtime']
    df_3h_back['hadmid_sample'] = df_3h_back['hadm_id'].astype(str) + '-3h'
    return df_3h_back

def first_not():
    print('正在处理正常患者 ')

    df_not_first = df_not.sort_values(by=['hadm_id', 'starttime'],ascending=[True, True])
    df_not_first['hadmid_sample'] = df_not_first['hadm_id'].astype(str) + 'first'
    df_not_first['ill_label'] = 0
    return df_not_first

def add_not():
    print('正在处理正常患者 中补充的部分 ')
    df_not_subject = set(df_not['hadm_id'])
    random_elements_to_remove = random.sample(df_not_subject, int(len(df_not_subject) // 2))
    for element in random_elements_to_remove:
        df_not_subject.remove(element)
    df_not_add = df_not[df_not['hadm_id'].isin(df_not_subject)]

    df_not_add_random_endtime = df_not_add.drop_duplicates(subset=['hadm_id'], keep='last')
    df_not_add_random_endtime['endtime'] = pd.to_datetime(df_not_add_random_endtime['endtime'])
    df_not_add_random_endtime['random_endtime'] = df_not_add_random_endtime['endtime']-pd.Timedelta(hours=3)

    df_not_add = pd.merge(df_not_add, df_not_add_random_endtime[['hadm_id', 'random_endtime']], on=['hadm_id'], how='left')
    df_not_add['endtime'] = pd.to_datetime(df_not_add['endtime'])
    df_not_add = df_not_add[df_not_add['random_endtime'] >= df_not_add['endtime']]

    df_not_add = df_not_add.sort_values(by=['hadm_id', 'starttime'], ascending=[True, True])
    df_not_add['hadmid_sample'] = df_not_add['hadm_id'].astype(str) + 'second'
    df_not_add['ill_label'] = 0
    del df_not_add['random_endtime']
    return df_not_add

def add_not_12h():
    print('正在处理正常患者 12h ')
    df_12h_endtime = df_ill[df_ill['period'] == '12h']
    df_12h_endtime = df_12h_endtime.sample(frac=1, random_state=42)
    df_12h_endtime = df_12h_endtime.drop_duplicates(subset=['hadm_id'], keep='first')
    df_12h_endtime['12h_endtime'] = df_12h_endtime['endtime']
    df_12h = pd.merge(df_ill, df_12h_endtime[['hadm_id', '12h_endtime']], on=['hadm_id'], how='left')
    df_12h['12h_endtime'] = pd.to_datetime(df_12h['12h_endtime'])
    df_12h['endtime'] = pd.to_datetime(df_12h['endtime'])
    df_12h = df_12h[df_12h['12h_endtime'] >= df_12h['endtime']]
    df_12h = df_12h.sort_values(by=['hadm_id', 'starttime'],ascending=[True, True])
    df_12h['hadmid_sample'] = df_12h['hadm_id'].astype(str) + '12h'
    df_12h['ill_label'] = 0
    del df_12h['12h_endtime']
    return df_12h



#分训练集 验证集 测试集

def train_val_test(df_ill_3h,df_ill_0h,df_ill_3h_back,df_not,df_not_add,df_not_add_12h):
    df = pd.concat([df_ill_3h,df_ill_0h,df_ill_3h_back,df_not,df_not_add,df_not_add_12h])

    train_hadm_id = pd.read_csv(ROOT+'06-lstm_train_hadm_id.csv')
    val_hadm_id = pd.read_csv(ROOT+'07-lstm_val_hadm_id.csv')
    test_hadm_id = pd.read_csv(ROOT+'08-lstm_test_hadm_id.csv')

    df_train_input = df[df['hadm_id'].isin(set(train_hadm_id['hadm_id'].astype(int)))]
    df_val_input = df[df['hadm_id'].isin(set(val_hadm_id['hadm_id'].astype(int)))]
    df_test_input = df[df['hadm_id'].isin(set(test_hadm_id['hadm_id'].astype(int)))]

    #平衡正负样本
    # df_train_input = balance(df_train_input)
    # df_val_input = balance(df_val_input)
    # df_test_input = balance(df_test_input)
    print('==========TRAIN========')
    get_1_0_num(df_train_input)
    print('==========VAL========')
    get_1_0_num(df_val_input)
    print('==========TEST========')
    get_1_0_num(df_test_input)

    # 填充为336步长
    df_train_input = remove_data(df_train_input)
    df_val_input = remove_data(df_val_input)
    df_test_input = remove_data(df_test_input)
    if not os.path.exists(ROOT + f'{kind}_del_missdata_{miss_rate}'):
        os.makedirs(ROOT + f'{kind}_del_missdata_{miss_rate}')
    process_model_input(df_train_input, ROOT + f'{kind}_del_missdata_{miss_rate}/lstm_3h_train_model_input_data_by_{kind}_del_missdata_{miss_rate}.csv')
    process_model_input(df_val_input, ROOT + f'{kind}_del_missdata_{miss_rate}/lstm_3h_val_model_input_data_by_{kind}_del_missdata_{miss_rate}.csv')
    process_model_input(df_test_input, ROOT + f'{kind}_del_missdata_{miss_rate}/lstm_3h_test_model_input_data_by_{kind}_del_missdata_{miss_rate}.csv')


def balance(df):
    df_ill = df[df['ill_label'] == 1]
    df_not = df[df['ill_label'] == 0]
    df_ill_subject = df_ill[['hadmid_sample']].drop_duplicates(subset=['hadmid_sample'])
    df_not_subject = df_not[['hadmid_sample']].drop_duplicates(subset=['hadmid_sample'])
    if len(df_ill_subject) > len(df_not_subject):
        df_ill_subject = df_ill_subject.sample(len(df_not_subject))
        df_ill = df_ill[df_ill['hadmid_sample'].isin(set(df_ill_subject['hadmid_sample']))]
    else:
        df_not_subject = df_not_subject.sample(len(df_ill_subject))
        df_not = df_not[df_not['hadmid_sample'].isin(set(df_not_subject['hadmid_sample']))]

    df = pd.concat([df_ill,df_not])
    return df

def get_1_0_num(df):
    df_ill = df[df['ill_label'] == 1]
    df_not = df[df['ill_label'] == 0]

    df_ill_subject = df_ill[['hadm_id']].drop_duplicates(subset=['hadm_id'])
    df_not_subject = df_not[['hadm_id']].drop_duplicates(subset=['hadm_id'])
    print(f'患病患者数量 {len(df_ill_subject)}')
    print(f'正常患者数量 {len(df_not_subject)}')

    df_ill_subject = df_ill[['hadmid_sample']].drop_duplicates(subset=['hadmid_sample'])
    df_not_subject = df_not[['hadmid_sample']].drop_duplicates(subset=['hadmid_sample'])
    print(f'正样本hadmid_sample数量 {len(df_ill_subject)}')
    print(f'负样本hadmid_sample数量 {len(df_not_subject)}')

if __name__ == '__main__':

    train_val_test(time_period_3h_label(),time_period_0h_label(),time_period_3h_back_label(),first_not(),add_not(),add_not_12h())
    print('end')
