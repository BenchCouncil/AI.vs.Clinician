# -*- coding: utf-8 -*-
import pandas as pd
import random
import os
from datetime import datetime, timedelta
from multiprocessing import Pool
import warnings
import csv
import ast
import re

warnings.filterwarnings("ignore")

PATH = '/home/ddcui/'
ROOT = f'{PATH}mimic-original-data/mimiciv_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimic-lstm/data/mimiciv_preprocess/'

def get_mimiciv_sample_chart():
    chart_list = ['arrythmia', 'Blood Pressure systolic', 'GCS', 'Arterial pH', 'Respiratory Rate', 'PaO2', 'BUN',
                  'Heart Rate', 'FiO2', 'WBC', 'Platelets', 'Arrythmia', 'Alanine Aminotransferase',
                  'Amylase', 'Asparate Aminotransferase', 'Bicarbonate', 'Blood Pressure Diastolic',
                  'Hematocrit', 'Hemoglobin', 'Lactate', 'Lipase', 'Oxygen saturation', 'Potassium',
                  'Sodium', 'Temperature', 'Creatinine', 'Bilirubin', 'Weight', 'Arterial PaCO2',
                  'MAP', 'weight', 'Arterial PaCO2', 'PAO2', 'CAM-ICU RASS LOC']

    df_sample = pd.read_csv(TO_ROOT + '10-mimiciv_3000_sample.csv',encoding='gbk',usecols=['hadm_id'])
    # df_sample = pd.read_csv(ROOT + 'mimiciv_preprocess/13-mimiciv_train_5_4_from_admittime.csv',encoding='gbk',usecols=['hadm_id'])

    df_item = pd.read_csv(ROOT + 'd_items.csv', usecols=['itemid', 'label'])
    missing_values0 = df_item['label'].isna()
    df_item = df_item[~missing_values0]
    df_labitem = pd.read_csv(ROOT + 'd_labitems.csv', usecols=['itemid', 'label'])
    missing_values1 = df_labitem['label'].isna()
    df_labitem = df_labitem[~missing_values1]

    columns = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valueuom']
    print("reading df_chartevents")
    for i, df_chunk in enumerate(
            pd.read_csv(ROOT + 'chartevents.csv', usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_chunk, df_item, on=['itemid'], how='inner')
        df_chunk = df_chunk[
            df_chunk[['label']].apply(lambda x: x.str.contains('|'.join(chart_list), case=False)).any(
                axis=1)]
        df_chunk = pd.merge(df_chunk,df_sample,on=['hadm_id'],how='inner')
        to_csv(df_chunk,TO_ROOT+'lstm_mimiciv_sample_subject_tscore_charts.csv')
        # to_csv(df_chunk,TO_ROOT+'lstm_mimiciv_train_subject_tscore_charts.csv')


    print("reading df_labevents")
    for i, df_lab_chunk in enumerate(
            pd.read_csv(ROOT + 'labevents.csv', usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_lab_chunk, df_labitem, on=['itemid'], how='inner')
        df_chunk = df_chunk[
            df_chunk[['label']].apply(lambda x: x.str.contains('|'.join(chart_list), case=False)).any(
                axis=1)]
        df_chunk = pd.merge(df_chunk,df_sample,on=['hadm_id'],how='inner')
        to_csv(df_chunk,TO_ROOT+'lstm_mimiciv_sample_subject_tscore_charts.csv')
        # to_csv(df_chunk,TO_ROOT+'lstm_mimiciv_train_subject_tscore_charts.csv')


def to_csv(df_chunk,filename):
    df_chunk = df_chunk.drop_duplicates()
    df_chunk = df_chunk.rename(columns={'subject_id': 'SUBJECT_ID', 'hadm_id': 'HADM_ID','itemid':'ITEMID','label':'LABEL','charttime':'CHARTTIME','value':'VALUE','valueuom':'VALUEUOM'})
    df_chunk = df_chunk.reindex(columns=['SUBJECT_ID','HADM_ID','ITEMID','LABEL','CHARTTIME','VALUE','VALUEUOM'])
    if not os.path.exists(filename):
        df_chunk.to_csv(filename, mode='w', index=False)
    else:
        df_chunk.to_csv(filename, mode='a', header=False, index=False)

if __name__ == '__main__':
    get_mimiciv_sample_chart()
    print('end')
