import numpy as np
import pandas as pd
import os
import uuid
from datetime import datetime, timedelta
import os
import csv
from multiprocessing import Pool


PATH = '/home/ddcui/'
ROOT_MIMICIV = f'{PATH}mimic-original-data/mimiciv_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'

def get_cxr_time(study_date,study_time):
    date_str = str(study_date)
    time_str = str(study_time)
    # 解析日期和时间
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    if study_time < 10:
        time_obj = datetime.strptime(time_str, "%S.%f")
    elif study_time < 100:
        time_obj = datetime.strptime(time_str, "%M%S.%f")
    else:
        time_obj = datetime.strptime(time_str, "%H%M%S.%f")
    combined_datetime = date_obj + timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second)
    return combined_datetime

def set_cxt_datetime():
    df_redio = pd.read_csv(TO_ROOT+'preprocess/mimic-cxr-2.0.0-metadata.csv',encoding='gbk')

    df_redio['studytime'] = ''
    for index, row in df_redio.iterrows():
        study_date = row['StudyDate']
        study_time = row['StudyTime']
        study_datatime = pd.to_datetime(get_cxr_time(study_date, round(study_time, 3)))
        df_redio.at[index, 'studytime'] = study_datatime
    df_redio.to_csv(TO_ROOT+'preprocess/mimic-cxr-2.0.0-metadata_plus_studytime.csv',encoding='gbk',index=False)

def ill_not_cxr(kind):
    df_sample = pd.read_csv(TO_ROOT + f'preprocess/1-mimiciv_{kind}_redio_subject_data_distribute.csv', encoding='gbk')
    df_redio = pd.read_csv(TO_ROOT+ 'preprocess/mimic-cxr-2.0.0-metadata_plus_studytime.csv', encoding='gbk')
    df_sample['starttime'] = df_sample['start_endtime'].str.split('~').str[0]
    df_sample['endtime'] = df_sample['start_endtime'].str.split('~').str[1]
    df_redio = df_redio[df_redio['subject_id'].isin(set(df_sample['subject_id']))]
    df_redio = pd.merge(df_redio, df_sample, on=['subject_id'], how='left')
    df_redio['admittime'] = pd.to_datetime(df_redio['admittime'])
    df_redio = df_redio[df_redio['studytime'] != '']
    df_redio_group = df_redio.groupby('subject_id')
    pool = Pool(processes=20)
    pool.map(process_subject, [(subject_id, group_data,kind) for subject_id, group_data in df_redio_group])
    pool.close()
    pool.join()

def process_subject(args):
    subject_id, group_data,kind = args
    df_cxr = pd.DataFrame()
    for index, row in group_data.iterrows():
        studytime = pd.to_datetime(row['studytime'])
        admittime = pd.to_datetime(row['admittime'])
        endtime = pd.to_datetime(row['endtime'])
        if studytime >= admittime and studytime <= endtime + pd.Timedelta(days=7):
            df_cxr = pd.concat([df_cxr, row.to_frame().T])
    if len(df_cxr) > 0:
        if kind == 'ill':
            columns_to_keep = ['subject_id', 'morta_90', 'ill_time', 'hadm_id', '性别', '年龄', 'admittime', '体重', 'time_range',
                           'start_endtime', 'base_current_sum', 'next_current_sum', 'is_have_qsofa', 'qsofa_score',
                           'starttime', 'endtime']
        else:
            columns_to_keep = ['subject_id', 'hadm_id', '性别', '年龄', 'admittime', '体重',
                               'time_range',
                               'start_endtime', 'base_current_sum', 'next_current_sum', 'is_have_qsofa', 'qsofa_score',
                               'starttime', 'endtime']
        df_cxr = df_cxr[columns_to_keep]
        df_cxr = df_cxr.drop_duplicates()
        filename = TO_ROOT + f'preprocess/1-mimiciv_{kind}_redio_subject_data_distribute_havecxr.csv'
        if not os.path.exists(filename):
            df_cxr.to_csv(filename, mode='w', index=False, quoting=csv.QUOTE_ALL, encoding='gbk')
        else:
            df_cxr.to_csv(filename, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL, encoding='gbk')



if __name__ == "__main__":
    print('转化mimiciv-cxr中时间')
    set_cxt_datetime()
    print('筛选存在cxr的患者')
    ill_not_cxr('ill')
    ill_not_cxr('not')
    print('end')


