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
ROOT = f'{PATH}mimic-original-data/mimiciii_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimic-lstm/data/mimiciii_preprocess/'

def get_tscore_all_charts():
    chart_list = ['arrythmia', 'Blood Pressure systolic', 'GCS', 'Arterial pH', 'Respiratory Rate', 'PaO2', 'BUN',
                  'Heart Rate', 'FiO2', 'WBC', 'Platelets', 'Arrythmia', 'Alanine Aminotransferase',
                  'Amylase', 'Asparate Aminotransferase', 'Bicarbonate', 'Blood Pressure Diastolic',
                  'Hematocrit', 'Hemoglobin', 'Lactate', 'Lipase', 'Oxygen saturation', 'Potassium',
                  'Sodium', 'Temperature', 'Creatinine', 'Bilirubin', 'Weight', 'Arterial PaCO2',
                  'MAP', 'weight', 'Arterial PaCO2', 'PAO2', 'CAM-ICU RASS LOC']

    df_item = pd.read_csv(ROOT + 'D_ITEMS.csv', usecols=['ITEMID', 'LABEL'])
    missing_values0 = df_item['LABEL'].isna()
    df_item = df_item[~missing_values0]
    df_labitem = pd.read_csv(ROOT + 'D_LABITEMS.csv', usecols=['ITEMID', 'LABEL'])
    missing_values1 = df_labitem['LABEL'].isna()
    df_labitem = df_labitem[~missing_values1]

    columns = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUEUOM']
    print("reading df_chartevents")
    for i, df_chunk in enumerate(
            pd.read_csv(ROOT + 'CHARTEVENTS.csv', usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_chunk, df_item, on=['ITEMID'], how='inner')
        df_chunk = df_chunk[
            df_chunk[['LABEL']].apply(lambda x: x.str.contains('|'.join(chart_list), case=False)).any(
                axis=1)]
        filename = TO_ROOT + 'tscore_model_all_chart.csv'
        if not os.path.exists(filename):
            df_chunk.to_csv(filename, mode='w', index=False)
        else:
            df_chunk.to_csv(filename, mode='a', header=False, index=False)

    print("reading df_labevents")
    for i, df_lab_chunk in enumerate(
            pd.read_csv(ROOT + 'LABEVENTS.csv', usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_lab_chunk, df_labitem, on=['ITEMID'], how='inner')
        df_chunk = df_chunk[
            df_chunk[['LABEL']].apply(lambda x: x.str.contains('|'.join(chart_list), case=False)).any(
                axis=1)]
        filename = TO_ROOT + 'tscore_model_all_chart.csv'
        if not os.path.exists(filename):
            df_chunk.to_csv(filename, mode='w', index=False)
        else:
            df_chunk.to_csv(filename, mode='a', header=False, index=False)


def get_sample_chart():
    df_sepsis = pd.read_csv(ROOT + '09-tscore_train_val_test_ill_subject.csv', usecols=['HADM_ID'], encoding='gbk')
    df_not = pd.read_csv(ROOT + '09-tscore_train_val_test_not_subject.csv',usecols=[ 'HADM_ID'])
    df_tscore_charts = pd.read_csv(TO_ROOT + 'tscore_model_all_chart.csv')


    df_sepsis_charts = pd.merge(df_sepsis, df_tscore_charts, on=['HADM_ID'], how='inner')
    df_not_charts = pd.merge(df_not, df_tscore_charts, on=['HADM_ID'], how='inner')
    df_sepsis_charts.to_csv(ROOT + 'tscore_train_val_test_ill_subject_tscore_charts.csv', mode='w',
                                 index=False)
    df_not_charts.to_csv(ROOT + 'tscore_train_val_test_not_subject_tscore_charts.csv', mode='w',
                               index=False)


if __name__ == '__main__':
    get_tscore_all_charts()
    get_sample_chart()
    print('end')
