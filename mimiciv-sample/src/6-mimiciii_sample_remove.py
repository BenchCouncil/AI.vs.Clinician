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
from datetime import datetime


PATH = '/home/ddcui/'
ROOT_MIMICIV = f'{PATH}mimic-original-data/mimiciv_database/'
ROOT_MIMICIII = f'{PATH}mimic-original-data/mimiciii_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'


def mimiciii_remove_sample():
    #先选出mimiciv中sample中可能与mimiciii重合的患者
    df_sample = pd.read_csv(TO_ROOT+ 'preprocess/10-mimiciv_3000_sample.csv',usecols=['subject_id','hadm_id','admittime','ill_time'],encoding='gbk')
    df_patient = pd.read_csv(ROOT_MIMICIV + 'patients.csv',usecols=['subject_id','gender','anchor_age','anchor_year_group'])
    df_patient['anchor_year_group'] = df_patient['anchor_year_group'].astype(str).str.split(' - ').str[0]
    df_patient['anchor_year_group'] = df_patient['anchor_year_group'].astype(int)
    df_sample = pd.merge(df_sample,df_patient,on=['subject_id'],how='inner')
    df_sample = df_sample.drop_duplicates()
    df_sample = df_sample[df_sample['anchor_year_group'] < 2013]

    df_admission = pd.read_csv(ROOT_MIMICIV + 'admissions.csv',usecols=['hadm_id','dischtime','marital_status','ethnicity'])
    df_sample = pd.merge(df_sample,df_admission,on=['hadm_id'],how='inner')
    df_sample['admittime'] = pd.to_datetime(df_sample['admittime'])
    # df_sample['admittime'] = pd.to_datetime(df_sample['admittime'],format='mixed')
    df_sample['dischtime'] = pd.to_datetime(df_sample['dischtime'])
    df_sample['distance'] = df_sample['dischtime'] - df_sample['admittime']


    df_patinet_iii = pd.read_csv(ROOT_MIMICIII + 'PATIENTS.csv',usecols=['SUBJECT_ID','GENDER','DOB'])
    df_admission_iv = pd.read_csv(ROOT_MIMICIII + 'ADMISSIONS.csv',usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','MARITAL_STATUS','ETHNICITY'])
    df_patinet_iii = pd.merge(df_patinet_iii,df_admission_iv,on=['SUBJECT_ID'],how='inner')
    df_patinet_iii['ADMITTIME'] = pd.to_datetime(df_patinet_iii['ADMITTIME'])
    df_patinet_iii['DISCHTIME'] = pd.to_datetime(df_patinet_iii['DISCHTIME'])
    df_patinet_iii['DOB'] = df_patinet_iii['DOB'].astype(str).str.split('-').str[0]
    df_patinet_iii['ADMITTIME_YEAR'] = df_patinet_iii['ADMITTIME'].astype(str).str.split('-').str[0]
    df_patinet_iii['AGE'] = df_patinet_iii['ADMITTIME_YEAR'].astype(int) - df_patinet_iii['DOB'].astype(int)
    del df_patinet_iii['ADMITTIME_YEAR']
    df_patinet_iii['distance'] = df_patinet_iii['DISCHTIME'] - df_patinet_iii['ADMITTIME']

    i = 0
    i_iii = 0
    df_result = pd.DataFrame()
    for index,row in df_sample.iterrows():
        #人口学信息一致   年龄信息可能不准确 先不用
        hadm_id = row['hadm_id']
        distance = row['distance']
        marital_status = row['marital_status']
        ethnicity =  row['ethnicity']
        gender = row['gender']
        df = df_patinet_iii[(df_patinet_iii['distance']==distance) & (df_patinet_iii['MARITAL_STATUS']==marital_status) & (df_patinet_iii['ETHNICITY']==ethnicity) & (df_patinet_iii['GENDER']==gender)]

        if len(df) > 0:
            print(f'sample中的患者hadm_id {hadm_id}')
            print(df)
            i_iii += len(df)
            df_result = pd.concat([df_result,df])
            i += 1

    print(f'sample人口学信息 可能重复的 样本中{i}')
    print(f'人口学信息 可能重复的 mimiiii中{i_iii}')
    print(len(df_result))

    df_result[['HADM_ID']].to_csv(TO_ROOT+ 'preprocess/3-mimiciv_3000_sample_duplicated_with_mimiciii.csv',index=False,mode='w')

if __name__ == '__main__':
    mimiciii_remove_sample()
    print('end')