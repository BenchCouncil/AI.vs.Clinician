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

PATH = '/home/ddcui/'
ROOT = f'{PATH}/hai-med-database/mimic-lstm/data/mimiciv_preprocess/'

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
    if 'current_period' in columns_list:
        del df['current_period']
    return df

kind = 'diff_meanvalue'

df_sample = pd.read_csv(ROOT+f'lstm_mimiciv_sample_model_input_data_by_{kind}.csv')
temp = df_sample.drop_duplicates(subset=['current_period'])
print(f'sample总数 {len(temp)}')

def sample_current_perdiod_label():
    print('正在处理sample的3h ')
    #打标签 并且排序
    df_not = df_sample[df_sample['illtime'].astype(str) == 'nan']
    df_ill = df_sample[df_sample['illtime'].astype(str) != 'nan']
    df_ill['illtime'] = pd.to_datetime(df_ill['illtime'])
    df_ill['endtime'] = pd.to_datetime(df_ill['endtime'])

    df_not['ill_label'] = 0
    df_ill['ill_label'] = 1

    df_sample_label = pd.concat([df_not,df_ill])
    df_sample_label = df_sample_label.sort_values(by=['current_period', 'starttime'] ,ascending=[True, True])
    df_sample_label['hadmid_sample'] = df_sample_label['hadm_id'].astype(str) +'-'+ df_sample_label['current_period'].astype(str)

    return df_sample_label


def sample_test(df):
    # 填充为336步长
    df_test_input = remove_data(df)
    if not os.path.exists(ROOT + f'diff_meanvalue_del_missdata_0.4'):
        os.makedirs(ROOT + f'diff_meanvalue_del_missdata_0.4')
    process_model_input(df_test_input,ROOT + f'diff_meanvalue_del_missdata_0.4/lstm_3h_mimiciv_sample_model_input_data_by_{kind}.csv')



if __name__ == '__main__':

    sample_test(sample_current_perdiod_label())
    print('end')
