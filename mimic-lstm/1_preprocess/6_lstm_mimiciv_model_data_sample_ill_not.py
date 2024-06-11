import pandas as pd
import random
import os
from datetime import datetime, timedelta
from multiprocessing import Pool
import warnings
import csv
import ast
import re
import numpy as np
import math

warnings.filterwarnings("ignore")

PATH = '/home/ddcui/'
ROOT = f'{PATH}mimic-original-data/mimiciv_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimic-lstm/data/mimiciv_preprocess/'

column_list = ['subject_id', 'hadm_id','admittime','illtime','current_period','starttime','endtime', 'chronic airway obstruction',
               'chronic bronchitis', 'chronic pancreatitis',
               'chronic kidney disease',
               'chronic liver disease', 'chronic pulmonary', 'diabetes', 'emphysema', 'end stage renal disease',
               'heart failure', 'immunodeficiency',
               'organ insufficiency', 'rena insufficiency',
               'biliary complaint', 'cardiac complaint', 'dementia complaint', 'fall complaint',
               'gastrointestinal bleed complaint', 'seizure complaint',
               'stroke complain',
               'gender', 'weight', 'alanine transaminase', 'amylase', 'arterial pH', 'aspartate aminotransferase',
               'bicarbonate', 'bun', 'bun/creatinine ratio',
               'dbp', 'fio2', 'gcs', 'heart rate', 'hematocrit', 'hemoglobin', 'lactate', 'lipase',
               'oxygen saturation', 'arterial PaCO2', 'pao2/fio2 ratio', 'map',
               'platelet count', 'potassium', 'rass',
               'respiratory rate', 'shock index', 'sodium', 'sbp', 'temperature', 'wbc']


df_tscore_charts = pd.read_csv(TO_ROOT + 'lstm_mimiciv_sample_subject_tscore_charts.csv')
df_tscore_charts.loc[:, 'CHARTTIME'] = pd.to_datetime(df_tscore_charts['CHARTTIME'])
df_tscore_charts['VALUE'] = df_tscore_charts['VALUE'].replace({'Yes': 1, 'No': 0})
df_tscore_charts['VALUE'] = df_tscore_charts['VALUE'].replace({'+': ''})
mask = ~df_tscore_charts['LABEL'].str.contains('gcs', case=False)
df_tscore_charts.loc[mask, 'VALUE'] = pd.to_numeric(df_tscore_charts.loc[mask, 'VALUE'], errors='coerce')
df_tscore_charts = df_tscore_charts.dropna(subset=['VALUE'])

result_csv = TO_ROOT + 'lstm_mimiciv_sample_model_input_data_by_diff_meanvalue.csv'

# 准备患病患者 和 正常患者
df_sample_subject = pd.read_csv(TO_ROOT + '10-mimiciv_3000_sample.csv',encoding='gbk')
df_sample_subject.rename(columns={'subject_id':'SUBJECT_ID','hadm_id':'HADM_ID','性别':'GENDER','admittime':'ADMITTIME','ill_time':'ILL_TIME','starttime':'STARTTIME','endtime':'ENDTIME'}, inplace=True)
df_admittime = pd.read_csv(ROOT+'admissions.csv',usecols=['hadm_id','deathtime'])
df_admittime.rename(columns={'hadm_id':'HADM_ID','deathtime':'DEATHTIME'}, inplace=True)
df_sample_subject = pd.merge(df_sample_subject,df_admittime,on=['HADM_ID'],how='left')
df_sample_subject['GENDER'] = df_sample_subject['GENDER'].replace({'男': 1, '女': 0})


# 诊断表
df_diagnose = pd.read_csv(ROOT + 'diagnoses_icd.csv', usecols=['subject_id','hadm_id','icd_code'])
df_diagnose = df_diagnose.rename(columns={'subject_id': 'SUBJECT_ID', 'hadm_id': 'HADM_ID', 'icd_code':'ICD9_CODE'})
df_d_icd = pd.read_csv(ROOT + 'd_icd_diagnoses.csv', usecols=['icd_code', 'long_title'])
df_d_icd = df_d_icd.rename(columns={'icd_code': 'ICD9_CODE', 'long_title': 'SHORT_TITLE'})
df_diagnose = pd.merge(df_diagnose, df_d_icd, on=['ICD9_CODE'], how='inner')
df_diagnose['SHORT_TITLE'] = df_diagnose['SHORT_TITLE'].str.lower()

#mimic iv的不良事件表
df_ae = pd.read_csv(TO_ROOT + 'mimiciv_adverse_events.csv')
df_ae = df_ae.rename(
    columns={'hadm_id': 'HADM_ID', 'charttime': 'CHARTTIME'})
df_ae = df_ae[~df_ae['CHARTTIME'].isna()]
df_ae.set_index(['HADM_ID'], inplace=True)

#正常患者中死亡患者的平均值
df_mean_death = pd.read_csv(TO_ROOT+'lstm_model_data_fill_meanvalue_tabel_normal_death.csv')
df_mean_death['period_to_admintime'] = df_mean_death['period_to_admintime'].astype(int)
#正常患者中非死亡患者的平均值
df_mean_nodeath = pd.read_csv(TO_ROOT+'lstm_model_data_fill_meanvalue_tabel_normal_nodeath.csv')
df_mean_nodeath['period_to_admintime'] = df_mean_nodeath['period_to_admintime'].astype(int)
#患病患者的平均值
df_ill_mean = pd.read_csv(TO_ROOT+'lstm_model_data_fill_meanvalue_tabel_ill.csv')
df_ill_mean['period_end_to_illtime'] = df_ill_mean['period_end_to_illtime'].astype(int)

def tscore_model_data():
    # 使用多进程
    pool = Pool(processes=10)
    pool.map(process_not_row, [(index, row, df_tscore_charts) for index, row in df_sample_subject.iterrows()])
    pool.close()
    pool.join()



# 先筛选该时间段的数据

def process_not_row(args):
    index, row, df_tscore_charts = args
    subject_id = int(row['SUBJECT_ID'])
    hadm_id = int(row['HADM_ID'])
    admit_time = pd.to_datetime(row['ADMITTIME'])
    death_time = pd.to_datetime(row['DEATHTIME'])
    curr_period_start_time = pd.to_datetime(row['STARTTIME'])
    curr_period_end_time = pd.to_datetime(row['ENDTIME'])
    ill_time = row['ILL_TIME']
    current_period = row['start_endtime']
    print(f'-----正在准备索引为{index} 患病患者 SUBJECT_ID {subject_id},HADM_ID {hadm_id}数据中-----')
    gender = row['GENDER']

    df_sub_chart = df_tscore_charts[
        (df_tscore_charts['SUBJECT_ID'] == int(subject_id)) & (df_tscore_charts['HADM_ID'] == int(hadm_id))]
    df_sub_diagnose = df_diagnose[
        (df_diagnose['SUBJECT_ID'] == int(subject_id)) & (df_diagnose['HADM_ID'] == int(hadm_id))]
    df_subject_diagnose_set = set(df_sub_diagnose['SHORT_TITLE'])

    temptime = admit_time

    while temptime < curr_period_end_time:
        start_time = temptime
        end_time = start_time + pd.Timedelta(hours=1)

        hour_to_admintime = int(round(((end_time - admit_time).total_seconds() / 3600)))
        hour_to_illtime = None
        if ill_time is not None and str(ill_time) != 'nan':
            ill_time = pd.to_datetime(ill_time)
            hour_to_illtime = int(round(((ill_time - end_time).total_seconds() / 3600)))


        flag = is_have_adverse_events(subject_id, hadm_id, end_time)
        df_chart_current_period = df_sub_chart[(df_sub_chart['CHARTTIME'] < end_time) & (
                df_sub_chart['CHARTTIME'] >= start_time)]
        df_12h = df_sub_chart[(df_sub_chart['CHARTTIME'] < (end_time + timedelta(hours=12))) & (
                df_sub_chart['CHARTTIME'] >= (start_time - timedelta(hours=12)))]
        df_2h = df_sub_chart[(df_sub_chart['CHARTTIME'] < (end_time + timedelta(hours=2))) & (
                df_sub_chart['CHARTTIME'] >= (start_time - timedelta(hours=2)))]
        get_subject_charts(subject_id, hadm_id, df_chart_current_period, df_subject_diagnose_set, flag, df_12h, df_2h,
                           gender, start_time, end_time, admit_time, hour_to_admintime,death_time,ill_time,hour_to_illtime,current_period)

        temptime = end_time


def is_have_adverse_events(subject_id, hadm_id, ill_time):
    if hadm_id in df_ae.index:
        filtered_rows = df_ae.loc[hadm_id, :]
        if not filtered_rows.empty:
            ae_time = filtered_rows['CHARTTIME']
            # ae_time = filtered_rows.iloc[hadm_id]["CHARTTIME"]
            ae_time = pd.to_datetime(ae_time)
            ill_time = pd.to_datetime(ill_time)
            if ill_time >= ae_time:
                return True
    return False

def get_subject_charts(subject_id, hadm_id, df_chart_current_period, df_subject_diagnose_set, ae_flag, df_12h, df_2h,
                       gender, start_time, end_time, admit_time,hour_to_admittime,death_time,ill_time,hour_to_illtime,current_period):
    # 在当前检查项的时间段中筛选每列的值
    diagnose_lowercase_set = {s.lower() for s in df_subject_diagnose_set}

    # print(diagnose_lowercase_set)
    def diagnose(chart_name):
        if any(chart_name in element for element in diagnose_lowercase_set):
            return 1
        return 0

    chronic_airway_obstruction = diagnose('chronic airway obstruction')
    chronic_bronchitis = diagnose('chronic bronchitis')
    chronic_pancreatitis = diagnose('chronic pancreatitis')
    chronic_kidney_disease = diagnose('chronic kidney disease')
    chronic_liver_disease = diagnose('chronic liver disease')
    chronic_pulmonary = diagnose('chronic pulmonary')
    diabetes = diagnose('diabetes')
    emphysema = diagnose('emphysema')
    end_stage_renal_disease = diagnose('end stage renal disease')
    heart_failure = diagnose('heart failure')
    immunodeficiency = diagnose('immunodeficiency')
    organ_insufficiency = diagnose('organ dysfunction')
    biliary_complaint = diagnose('biliary')
    rena_insufficiency = diagnose('rena insufficiency')
    cardiac_complaint = diagnose('cardiac complaint')
    dementia_complaint = diagnose('dementia')
    fall_complaint = diagnose('fall_complaint')
    gastrointestinal_bleed_complaint = diagnose('gastrointest hemorr|gastrointestinal hemorrhage')
    seizure_complaint = diagnose('seizure complaint')
    stroke_complain = diagnose('stroke complain')

    alanine = get_final_chart_value('Alanine Aminotransferase (ALT)', df_chart_current_period, ae_flag,
                                    df_12h, hour_to_admittime,'alanine transaminase',death_time,ill_time,hour_to_illtime)
    amylase = get_final_chart_value(['Amylase'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'amylase',death_time,ill_time,hour_to_illtime)
    art_ph = get_final_chart_value(['Arterial pH'], df_chart_current_period,  ae_flag, df_2h, hour_to_admittime,'arterial pH',death_time,ill_time,hour_to_illtime)
    aspartate = get_final_chart_value(['Asparate Aminotransferase (AST)'], df_chart_current_period,  ae_flag,
                                      df_12h, hour_to_admittime,'aspartate aminotransferase',death_time,ill_time,hour_to_illtime)
    bicarbon = get_final_chart_value(['Bicarbonate'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'bicarbonate',death_time,ill_time,hour_to_illtime)
    bun = get_final_chart_value(['BUN'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'bun',death_time,ill_time,hour_to_illtime)
    dbp = get_final_chart_value(['Blood Pressure Diastolic'], df_chart_current_period,  ae_flag, df_2h,
                                hour_to_admittime,'dbp',death_time,ill_time,hour_to_illtime)
    fio2 = get_final_chart_value(['FiO2'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'fio2',death_time,ill_time,hour_to_illtime)
    hr = get_final_chart_value(['Heart Rate'], df_chart_current_period,  ae_flag, df_2h, hour_to_admittime,'heart rate',death_time,ill_time,hour_to_illtime)
    hematocrit = get_final_chart_value(['Hematocrit'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'hematocrit',death_time,ill_time,hour_to_illtime)
    hemoglobin = get_final_chart_value(['Hemoglobin'], df_chart_current_period,  ae_flag, df_12h,hour_to_admittime,'hemoglobin',death_time,ill_time,hour_to_illtime)
    lactate = get_final_chart_value(['Lactate'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'lactate',death_time,ill_time,hour_to_illtime)
    lipase = get_final_chart_value(['Lipase'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'lipase',death_time,ill_time,hour_to_illtime)
    os1 = get_final_chart_value(['Oxygen saturation'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'oxygen saturation',death_time,ill_time,hour_to_illtime)
    platelet = get_final_chart_value(['Platelets'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'platelet count',death_time,ill_time,hour_to_illtime)
    potassium = get_final_chart_value(['Potassium'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'potassium',death_time,ill_time,hour_to_illtime)
    rass = get_final_chart_value(['CAM-ICU RASS LOC'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'rass',death_time,ill_time,hour_to_illtime)
    rr = get_final_chart_value(['Respiratory Rate'], df_chart_current_period,  ae_flag, df_2h, hour_to_admittime,'respiratory rate',death_time,ill_time,hour_to_illtime)
    creatinine = get_final_chart_value(['Creatinine'], df_chart_current_period,  ae_flag, df_12h,hour_to_admittime,'',death_time,ill_time,hour_to_illtime)
    sodium = get_final_chart_value(['Sodium'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'sodium',death_time,ill_time,hour_to_illtime)
    sbp = get_final_chart_value(['Blood Pressure systolic'], df_chart_current_period,  ae_flag, df_2h,
                                hour_to_admittime,'sbp',death_time,ill_time,hour_to_illtime)
    wbc = get_final_chart_value(['WBC'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'wbc',death_time,ill_time,hour_to_illtime)
    pao2 = get_final_chart_value(['PaO2'], df_chart_current_period,  ae_flag, df_12h, hour_to_admittime,'',death_time,ill_time,hour_to_illtime)
    map = get_final_chart_value(['MAP'], df_chart_current_period,  ae_flag, df_2h, hour_to_admittime,'map',death_time,ill_time,hour_to_illtime)
    paco2 = get_final_chart_value(['Arterial PaCO2'], df_chart_current_period,  ae_flag, df_12h,hour_to_admittime,'arterial PaCO2',death_time,ill_time,hour_to_illtime)
    temperature = get_final_chart_value(['Temperature '], df_chart_current_period,  ae_flag, df_2h,
                                        hour_to_admittime,'temperature',death_time,ill_time,hour_to_illtime)
    weight = get_final_chart_value(
        ['Daily Weight', 'Admission Weight', 'Previous Weight', 'Present Weight', 'Weight Kg'], df_chart_current_period,
         ae_flag, df_12h, hour_to_admittime,'weight',death_time,ill_time,hour_to_illtime)
    gcs = get_final_chart_value(['GCS'], df_chart_current_period, ae_flag, df_2h,hour_to_admittime,'gcs',death_time,ill_time,hour_to_illtime)

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
    row_list = [subject_id, hadm_id,str(admit_time),str(ill_time),current_period,str(start_time), str(end_time),
                chronic_airway_obstruction, chronic_bronchitis, chronic_pancreatitis,
                chronic_kidney_disease,
                chronic_liver_disease, chronic_pulmonary, diabetes, emphysema,
                end_stage_renal_disease, heart_failure,
                immunodeficiency, organ_insufficiency, rena_insufficiency, biliary_complaint,
                cardiac_complaint, dementia_complaint,
                fall_complaint, gastrointestinal_bleed_complaint, seizure_complaint,
                stroke_complain, gender,
                weight, alanine, amylase, art_ph, aspartate, bicarbon, bun, bun_cr, dbp, fio2, gcs, hr,
                hematocrit, hemoglobin, lactate, lipase, os1, paco2, p02_fio2, map, platelet,
                potassium, rass, rr, shock_index,
                sodium, sbp, temperature, wbc]

    df_result.loc[len(df_result)] = row_list
    print(f'{row_list}')

    if not os.path.exists(result_csv):
        df_result.to_csv(result_csv, mode='w', index=False)
    else:
        df_result.to_csv(result_csv, mode='a', header=False, index=False)
    return df_result


def get_final_chart_value(chart_name_list, df_chart_current_period, ae_flag, df_xxh, hour_to_admittime,chart_column_name,death_time,ill_time,hour_to_illtime):
    if any('temperature' in name.lower() for name in chart_name_list):
        mask = df_chart_current_period['LABEL'].str.contains('Temperature C', case=False)
        df_chart_current_period.loc[mask, 'VALUE'] = df_chart_current_period.loc[mask, 'VALUE'].apply(
            lambda x: (float(x) * 1.8 + 32) if pd.to_numeric(x, errors='coerce') else x
        )
    if any('weight' in name.lower() for name in chart_name_list):
        mask = df_chart_current_period['LABEL'].str.contains('lbs.', case=False)
        df_chart_current_period.loc[mask, 'VALUE'] = df_chart_current_period.loc[mask, 'VALUE'].apply(
            lambda x: float(x) * 0.4536 if pd.to_numeric(x, errors='coerce') else x
        )
    if any('gcs' in name.lower() for name in chart_name_list):
        df_chart_current_period = get_consciousness(df_chart_current_period)
        df_xxh = get_consciousness(df_xxh)
    result_value = chart_value(chart_name_list, df_chart_current_period, ae_flag, df_xxh)
    if result_value == 0:
        if str(ill_time) == 'nan' or ill_time is None:
            #正常患者的平均值补充
            result_value = second_fill_0_not(chart_column_name, hour_to_admittime, death_time)
        else:
            #患病患者的平均值补充
            result_value = second_fill_0_ill(chart_column_name, hour_to_illtime)
    try:
        float_value = float(result_value)
        return round(float_value, 2)
    except ValueError:
        return 0

def get_consciousness(df_period):
    mask = df_period['LABEL'].str.contains('|'.join(['GCS Total']), case=False)
    df_consciousness = df_period[mask]
    if len(df_consciousness) > 0:
        df_consciousness['VALUE'] = pd.to_numeric(df_consciousness['VALUE'], errors='coerce')
    else:
        gcs_label = ['GCS - Eye Opening', 'GCS - Verbal Response', 'GCS - Motor Response']
        mask = df_period['LABEL'].str.contains('|'.join(gcs_label), case=False)
        df_consciousness = df_period[mask]
        df_consciousness['VALUE'] = df_consciousness['VALUE'].map(gcs_dict)
        if len(df_consciousness) > 0 and (len(df_consciousness) % 3 == 0):
            df_consciousness = get_gcs(df_consciousness)
            df_consciousness['VALUE'] = pd.to_numeric(df_consciousness['VALUE'], errors='coerce')
    if len(df_consciousness) == 0:
        return pd.DataFrame(columns=df_period.columns.tolist())
    else:
        return df_consciousness

gcs_dict = {
    'To Pain': 2,
    'Spontaneously': 4,
    'To Speech': 3,
    'None': 1,

    'No Response': 1,
    'Inappropriate Words': 4,
    'Confused': 2,
    'Incomprehensible sounds': 3,
    'Oriented': 5,
    'No Response-ETT': 0,

    'Abnormal Flexion': 3,
    'Flex-withdraws': 4,
    'Obeys Commands': 6,
    'Abnormal extension': 2,
    'Localizes Pain': 5,
    'No response': 1
}

def get_gcs(df_gcs):
    grouped_df = df_gcs.groupby(['CHARTTIME'])
    for group_key, group_index in grouped_df.groups.items():
        group = grouped_df.get_group(group_key)
        count = sum(group['VALUE'])
        group_index_list = group_index.tolist()
        df_gcs.loc[group_index_list, 'VALUE'] = count
    df_gcs = df_gcs.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'VALUE'])
    df_gcs = df_gcs.sort_values('CHARTTIME', ascending=False)
    return df_gcs


# 找当前时间范围内的检查项 或者 根据不良事件和检查项周期进行补充  优先从补充左边的
def chart_value(chart_name_list, df_chart_current_period, ae_flag, df_xxh):
    chart_name_list = [name.lower() for name in chart_name_list]
    mask = df_chart_current_period['LABEL'].str.contains('|'.join(chart_name_list), case=False)
    df_chart = df_chart_current_period[mask]
    if len(df_chart) == 0:
        if ae_flag:
            return 0
        else:
            mask = df_xxh['LABEL'].str.contains('|'.join(chart_name_list), case=False)
            df_xxh_chart = df_xxh[mask]
            if len(df_xxh_chart) == 0:
                return 0
            else:
                df_xxh_chart = df_xxh_chart.sort_values('CHARTTIME', ascending=True)
                return df_xxh_chart.loc[df_xxh_chart['VALUE'].first_valid_index(), 'VALUE']
    else:
        return df_chart.loc[df_chart['VALUE'].first_valid_index(), 'VALUE']


# 患病患者  找距离患病时间相同时间段 所有患者的平均值
def second_fill_0_not(chart_column_name, hour_to_admittime,death_time):
    if chart_column_name != '':
        if death_time is None or death_time == '':
            #用非死亡表填充
            df_mean_subject = df_mean_nodeath[df_mean_nodeath['period_to_admintime'] == hour_to_admittime]
        else:
            #用死亡表填充
            df_mean_subject = df_mean_death[df_mean_death['period_to_admintime'] == hour_to_admittime]

        if len(df_mean_subject) > 0:
            value = df_mean_subject.iloc[0][chart_column_name]
            return float(value)
    return 0

# 患病患者  找距离患病时间相同时间段 所有患者的平均值
def second_fill_0_ill(chart_column_name, hour_to_illtime):
    if chart_column_name != '':
        df_ill_mean_subject = df_ill_mean[df_ill_mean['period_end_to_illtime'] == hour_to_illtime]
        if len(df_ill_mean_subject) > 0:
            value = df_ill_mean_subject.iloc[0][chart_column_name]
            return float(value)
    return 0


if __name__ == '__main__':
    tscore_model_data()
    print('end')
