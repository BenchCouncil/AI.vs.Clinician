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

column_list = ['weight', 'alanine transaminase', 'amylase', 'arterial pH', 'aspartate aminotransferase',
               'bicarbonate', 'bun', 'bun/creatinine ratio',
               'dbp', 'fio2', 'gcs', 'heart rate', 'hematocrit', 'hemoglobin', 'lactate', 'lipase',
               'oxygen saturation', 'arterial PaCO2', 'pao2/fio2 ratio', 'map',
               'platelet count', 'potassium', 'rass',
               'respiratory rate', 'shock index', 'sodium', 'sbp', 'temperature', 'wbc', 'period_to_admintime']


#TODO *****加载Tscore模型需要的检查项数据 这个是用来算正常患者的平均值用的固定文件******
df_tscore_charts = pd.read_csv(TO_ROOT + 'tscore_not_back6000_patients_sample_charts.csv')
df_tscore_charts.loc[:, 'CHARTTIME'] = pd.to_datetime(df_tscore_charts['CHARTTIME'])
df_tscore_charts['VALUE'] = df_tscore_charts['VALUE'].replace({'Yes': 1, 'No': 0})
df_tscore_charts['VALUE'] = df_tscore_charts['VALUE'].replace({'+': ''})
mask = ~df_tscore_charts['LABEL'].str.contains('gcs',case=False)
df_tscore_charts.loc[mask, 'VALUE'] = pd.to_numeric(df_tscore_charts.loc[mask, 'VALUE'], errors='coerce')
df_tscore_charts = df_tscore_charts.dropna(subset=['VALUE'])

df_admit = pd.read_csv(ROOT + 'ADMISSIONS.csv',usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','DEATHTIME'])
df_admit['ADMITTIME'] = pd.to_datetime(df_admit['ADMITTIME'])


#计算上述检查项 距离患病时间不同时间段的平均值 保存成映射表
def tscore_model_data():
    # 使用多进程
    pool = Pool(processes=5)
    pool.map(get_subject_charts, [index for index in range(0,337)])

    pool.close()
    pool.join()
    # for index, row in df_ill_subject.iterrows():
    #     process_ill_row(index, row, df_tscore_charts)


def get_subject_charts(time_period):
    # 在当前检查项的时间段中筛选每列的值

    alanine = get_final_chart_value('Alanine Aminotransferase (ALT)',time_period)
    amylase = get_final_chart_value(['Amylase'], time_period)
    art_ph = get_final_chart_value(['Arterial pH'], time_period)
    aspartate = get_final_chart_value(['Asparate Aminotransferase (AST)'], time_period)
    bicarbon = get_final_chart_value(['Bicarbonate'], time_period)
    bun = get_final_chart_value(['BUN'], time_period)
    dbp = get_final_chart_value(['Blood Pressure Diastolic'], time_period)
    fio2 = get_final_chart_value(['FiO2'], time_period)
    hr = get_final_chart_value(['Heart Rate'], time_period)
    hematocrit = get_final_chart_value(['Hematocrit'], time_period)
    hemoglobin = get_final_chart_value(['Hemoglobin'], time_period)
    lactate = get_final_chart_value(['Lactate'], time_period)
    lipase = get_final_chart_value(['Lipase'],time_period)
    os1 = get_final_chart_value(['Oxygen saturation'],time_period)
    platelet = get_final_chart_value(['Platelets'], time_period)
    potassium = get_final_chart_value(['Potassium'], time_period)
    rass = get_final_chart_value(['CAM-ICU RASS LOC'],time_period)
    rr = get_final_chart_value(['Respiratory Rate'], time_period)
    creatinine = get_final_chart_value(['Creatinine'], time_period)
    sodium = get_final_chart_value(['Sodium'], time_period)
    sbp = get_final_chart_value(['Blood Pressure systolic'], time_period)
    wbc = get_final_chart_value(['WBC'], time_period)
    pao2 = get_final_chart_value(['PaO2'], time_period)
    map = get_final_chart_value(['MAP'], time_period)
    paco2 = get_final_chart_value(['Arterial PaCO2'], time_period)
    temperature = get_final_chart_value(['Temperature '],time_period)
    weight = get_final_chart_value(
        ['Daily Weight', 'Admission Weight', 'Previous Weight', 'Present Weight', 'Weight Kg'], time_period)
    gcs = get_final_chart_value(['GCS Total'], time_period)

    p02_fio2 = 0
    if fio2 != 0:
        p02_fio2 = round(pao2 / fio2, 2)
    bun_cr = 0
    # print(creatinine)
    if creatinine > 0:
        bun_cr = round(bun / creatinine, 2)
    shock_index = 0
    if sbp != 0:
        shock_index = round((hr / sbp), 2)

    df_result = pd.DataFrame(columns=column_list)
    row_list = [ weight, alanine,
                amylase, art_ph, aspartate, bicarbon, bun, bun_cr, dbp, fio2, gcs, hr,
                hematocrit, hemoglobin, lactate, lipase, os1, paco2, p02_fio2, map, platelet,
                potassium, rass, rr, shock_index, sodium, sbp, temperature, wbc, time_period]

    df_result.loc[len(df_result)] = row_list

    result_csv = TO_ROOT+'lstm_model_data_fill_meanvalue_tabel_normal_death.csv'
    if not os.path.exists(result_csv):
        df_result.to_csv(result_csv, mode='w', index=False)
    else:
        print(f'{row_list}')
        df_result.to_csv(result_csv, mode='a', header=False, index=False)
    return df_result


def get_final_chart_value(chart_name_list, time_period):
    result_value = second_fill_0_death(chart_name_list, time_period)
    try:
        float_value = float(result_value)
        return round(float_value, 2)
    except ValueError:
        return 0

# 患病患者  找距离患病时间相同时间段 所有患者的平均值
def second_fill_0_death(chart_name_list, time_period):
    mask = df_tscore_charts['LABEL'].str.contains('|'.join(chart_name_list), case=False)
    df_all_subject_chart = df_tscore_charts[mask]
    df_all_subject_chart = pd.merge(df_admit, df_all_subject_chart, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    df_all_subject_chart = df_all_subject_chart[~df_all_subject_chart['DEATHTIME'].isna()]

    df_all_subject_chart = df_all_subject_chart[
        (df_all_subject_chart['CHARTTIME'] < (
                df_all_subject_chart['ADMITTIME'] + timedelta(hours=time_period) + timedelta(hours=1))) &
        (df_all_subject_chart['CHARTTIME'] >= df_all_subject_chart['ADMITTIME'] + timedelta(hours=time_period))]
    # print(chart_name_list)
    if len(df_all_subject_chart) == 0:
        return 0
    else:
        df_all_subject_chart['VALUE'] = pd.to_numeric(df_all_subject_chart['VALUE'], errors='coerce')
        count_num = df_all_subject_chart['VALUE'].sum() / len(df_all_subject_chart)
        return round(float(count_num), 2)



if __name__ == '__main__':
    tscore_model_data()
    print('end')
