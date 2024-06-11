import pandas as pd
import re
import os
import numpy as np
import ast
import copy
import csv
from datetime import datetime, timedelta
import multiprocessing
import csv
import random
import resource
import csv
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

PATH = '/home/ddcui/'
ROOT = f'{PATH}mimic-original-data/mimiciv_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'

TB_D_ITEM = 'd_items.csv'
TB_D_LAB = 'd_labitems.csv'
TB_D_DIAG = 'd_icd_diagnoses.csv'
TB_D_PROCED = 'd_icd_procedures.csv'
TB_CHART = 'chartevents.csv'
TB_LAB = 'labevents.csv'
TB_INPUT = 'inputevents.csv'
TB_OUTPUT = 'outputevents.csv'
TB_MICRO = 'microbiologyevents.csv'
TB_PROCE = 'procedureevents.csv'
TB_PRESCR = 'prescriptions.csv'
TB_PATIENT = 'patients.csv'
TB_ADMISS = 'admissions.csv'
TB_DIAG = 'diagnoses_icd.csv'
TB_PROCE_ICD = 'procedures_icd.csv'
TB_ICUSTAY = 'icustays.csv'
TB_DATETIME = 'datetimeevents.csv'

ITEMID = 'itemid'
LABEL = 'label'
CATEGORY = 'category'
SUBJECT_ID = 'subject_id'
HADM_ID = 'hadm_id'
ICU_ID = 'stay_id'

CHARTTIME = 'charttime'
VALUE = 'value'
VALUENUM = 'valuenum'
VALUEUOM = 'valueuom'
STARTTIME = 'starttime'
AMOUNT = 'amount'
AMOUNTUOM = 'amountuom'
TOTALAMOUNT = 'totalamount'
RATE = 'rate'
RATEUOM = 'rateuom'
SPEC_TYPE_DESC = 'spec_type_desc'
ORG_NAME = 'org_name'
ISOLATE_NUM = 'isolate_num'
AB_ITEMID = 'ab_itemid'
AB_NAME = 'ab_name'
DILUTION_TEXT = 'dilution_text'
DILUTION_COMPARISON = 'dilution_comparison'
DILUTION_VALUE = 'dilution_value'
INTERPRETATION = 'interpretation'
ICD9_CODE = 'icd_code'
LONG_TITLE = 'long_title'
ADMITTIME = 'admittime'
STARTDATE = 'startdate'
DRUG = 'drug'
LINKSTO = 'linksto'
ILL_TIME = 'ill_time'
GENDER = 'gender'
AGE = 'anchor_age'
YEAR_GROUP = 'anchor_year_group'
STUDYDATE = 'StudyDate'
DICOMID = 'dicom_id'
MA_STATUS = 'marital_status'
ETHNICITY = 'ethnicity'
MORTA_90d ='morta_90'


df_charts = pd.read_csv(TO_ROOT + 'preprocess/mimiciv_ill_redio_subject_doctor_chart_lab.csv',
    usecols=[SUBJECT_ID,HADM_ID, CHARTTIME, VALUE,VALUENUM, VALUEUOM,LABEL])
df_charts[HADM_ID] = df_charts[HADM_ID].astype(str).str.strip().replace('\.0', '', regex=True)
df_charts[SUBJECT_ID] = df_charts[SUBJECT_ID].astype(int)
df_charts[HADM_ID] = df_charts[HADM_ID].astype(int)
df_charts.set_index([HADM_ID], inplace=True)
df_charts = df_charts[(~df_charts[VALUE].isna()) | (~df_charts[VALUE].isnull()) | ((df_charts[VALUE] != ''))]

df_item = pd.read_csv(ROOT +TB_D_ITEM, usecols=[ITEMID,LABEL, CATEGORY])

df_input = pd.read_csv(ROOT + TB_INPUT,usecols=[SUBJECT_ID,HADM_ID,STARTTIME, ITEMID,AMOUNT,AMOUNTUOM, TOTALAMOUNT,RATE,RATEUOM])
df_input[AMOUNT] = pd.to_numeric(df_input[AMOUNT], errors='coerce').astype(float)
df_input = df_input.dropna(subset=[AMOUNT], how='any')
df_input.rename(columns={STARTTIME: CHARTTIME}, inplace=True)
df_input = pd.merge(df_input, df_item, on=[ITEMID], how='inner')
df_input.set_index([HADM_ID], inplace=True)

df_output = pd.read_csv(ROOT + TB_OUTPUT,
                        usecols=[SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE, VALUEUOM])
df_output = pd.merge(df_output, df_item, on=[ITEMID], how='inner')
df_output.set_index([HADM_ID], inplace=True)
df_output[VALUE] = pd.to_numeric(df_output[VALUE], errors='coerce').astype(float)
df_output = df_output.dropna(subset=[VALUE], how='any')

df_microbiology = pd.read_csv(ROOT + TB_MICRO,usecols=[SUBJECT_ID, HADM_ID, CHARTTIME, SPEC_TYPE_DESC, ORG_NAME,ISOLATE_NUM, AB_ITEMID, AB_NAME, DILUTION_TEXT, DILUTION_COMPARISON, DILUTION_VALUE, INTERPRETATION])
df_microbiology.set_index([HADM_ID], inplace=True)

df_icd_diagnose = pd.read_csv(ROOT + TB_D_DIAG, usecols=[ICD9_CODE, LONG_TITLE])
df_diagnose = pd.read_csv(ROOT + TB_DIAG, usecols=[SUBJECT_ID, HADM_ID, ICD9_CODE])
df_diagnose = pd.merge(df_diagnose, df_icd_diagnose, on=[ICD9_CODE], how='inner')
df_diagnose = df_diagnose.drop_duplicates()
df_diagnose.set_index([HADM_ID], inplace=True)

df_procedure = pd.read_csv(ROOT + TB_PROCE,
                              usecols=[SUBJECT_ID, HADM_ID, STARTTIME, ITEMID, VALUE, VALUEUOM])
df_procedure.rename(columns={STARTTIME: CHARTTIME}, inplace=True)
df_procedure = pd.merge(df_procedure, df_item, on=[ITEMID], how='inner')
df_procedure.set_index([HADM_ID], inplace=True)
df_procedure = df_procedure[
    (~df_procedure[VALUE].isna()) | (~df_procedure[VALUE].isnull()) | ((df_procedure[VALUE] != ''))]

df_ae = pd.read_csv(TO_ROOT + 'preprocess/mimiciv_adverse_events.csv')
df_ae = df_ae[~df_ae[CHARTTIME].isna()]
df_ae.set_index([HADM_ID], inplace=True)

def get_endtime_before(df, subject_id, hadm_id, endtime):
    if hadm_id in df.index:
        df = df.loc[hadm_id, :]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).T
    else:
        return pd.DataFrame(columns=df.columns)
    if CHARTTIME not in df.columns:
        return pd.DataFrame(columns=df.columns)
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]
    df[CHARTTIME] = pd.to_datetime(df[CHARTTIME], format=formats[0], errors='coerce')
    rows_to_convert = df[CHARTTIME].isnull()
    df.loc[rows_to_convert, CHARTTIME] = pd.to_datetime(df.loc[rows_to_convert, CHARTTIME], format
    =formats[1], errors='coerce')
    df = df.reset_index(drop=True)
    df[CHARTTIME] = pd.to_datetime(df[CHARTTIME])
    if endtime != '':
        df = df[df[CHARTTIME] <= pd.to_datetime(endtime)]
    df = df.sort_values(CHARTTIME, ascending=False)
    return df

def get_start_to_endtime(df, subject_id, hadm_id, starttime, endtime):
    df = df[(df["SUBJECT_ID"] == int(subject_id)) & (df["HADM_ID"] == int(hadm_id))]
    df = df.dropna(subset=[CHARTTIME])
    df[CHARTTIME] = pd.to_datetime(df[CHARTTIME])
    df = df[(df[CHARTTIME] < pd.to_datetime(endtime)) & (df[CHARTTIME] >= pd.to_datetime(starttime))]
    df = df.sort_values(CHARTTIME, ascending=False)
    return df

def get_batches(df):
    grouped = df.groupby(CHARTTIME)
    group_indexes = list(grouped.groups.values())
    grouped_dfs = [df.loc[indexes] for indexes in group_indexes]
    i = 0
    while i < len(grouped_dfs) - 1:
        current_data = grouped_dfs[i]
        next_data = grouped_dfs[i + 1]
        current_data = current_data.fillna('')
        next_data = next_data.fillna('')
        current_labels = set(current_data[LABEL].unique())
        next_labels = set(next_data[LABEL].unique())
        is_disjoint = len(current_labels.intersection(next_labels)) == 0
        if is_disjoint:
            merged_data = pd.concat([current_data, next_data])
            group_indexes[i] = merged_data.index
            grouped_dfs[i] = merged_data
            group_indexes.pop(i + 1)
            grouped_dfs.pop(i + 1)
        else:
            i += 1
    return grouped_dfs

def get_time_series(df, chartname, df_diagnose=None, illtime=''):
    time_series_list = []
    if isinstance(df, list):
        if not df:
            return None
    else:
        if df.empty and (df_diagnose is None or df_diagnose.empty):
            return None
        else:
            df = df.fillna('')
            # df[VALUEUOM] = df[VALUEUOM].astype(str).replace('nan', '')
    if chartname in ['体温', '体重', '意识']:
        df['temp'] = df.apply(
            lambda row: {'时间': str(row[CHARTTIME]), '值': str(row[VALUE]) + str(row[VALUEUOM]).replace('?', '')},
            axis=1)
        time_series_list = df['temp'].tolist()
        del df['temp']
    elif chartname == '影像报告':
        print('影像报告后续添加')
        # df['temp'] = df.apply(
        #     lambda row: {'影像类型': str(row['DESCRIPTION']), '时间': str(row[CHARTTIME]), '值': str(row['TEXT'])},
        #     axis=1)
        # time_series_list = df['temp'].tolist()
        # del df['temp']
    elif chartname == '血压':
        grouped = df.groupby(CHARTTIME)
        time_series_list1 = []
        for name, group in grouped:
            systolic_value = group.loc[group[LABEL].str.contains('systolic', case=False), VALUE].values
            diastolic_value = group.loc[group[LABEL].str.contains('diastolic', case=False), VALUE].values
            if len(systolic_value) == 0 or len(diastolic_value) == 0:
                return None
            entry = {'时间': str(name),
                     '值': str(int(float(systolic_value[0]))) + '/' + str(int(float(diastolic_value[0]))) + 'mmHg'}
            time_series_list1.append(entry)
        time_series_list = time_series_list1[::-1]
    elif chartname == '收缩压':
        df = df[df[VALUE] != '']
        df[VALUE] = pd.to_numeric(df[VALUE], errors='coerce')
        df = df.dropna(subset=[VALUE])
        df = df.reset_index(drop=True)
        df['temp'] = df.apply(
            lambda row: {'时间': str(row[CHARTTIME]), '值': str(int(float(row[VALUE]))) + row[VALUEUOM]},
            axis=1)
        time_series_list = df['temp'].tolist()
        del df['temp']
    elif chartname in ['心率', '呼吸频率']:
        df = df[df[VALUE] != '']
        df[VALUE] = pd.to_numeric(df[VALUE], errors='coerce')
        df = df.dropna(subset=[VALUE])
        df = df.reset_index(drop=True)
        df['temp'] = [{'时间': str(ct), '值': str(int(float(v))) + uom} for ct, v, uom in
                      zip(df[CHARTTIME], df[VALUE], df[VALUEUOM])]

        # df['temp'] = df.apply(
        #     lambda row: {'时间': str(row[CHARTTIME]), '值': str(int(float(row[VALUE]))) + row[VALUEUOM]}, axis=1)
        time_series_list = df['temp'].tolist()
        del df['temp']
    elif chartname == '输入':
        df = df[df[AMOUNT] != 0.0]
        df = df.dropna(subset=[AMOUNT], how='any')
        grouped = df.groupby(CHARTTIME)
        time_series_list1 = []
        for name, group in grouped:
            group['temp1'] = group.apply(
                lambda row: {key: value for key, value in {
                    '入量': str(row[AMOUNT]) + str(row[AMOUNTUOM]),
                    '给药速率': str(row[RATE]) + str(row[RATEUOM]),
                    '补液名称': str(row[LABEL]),
                    '补液类型': row[CATEGORY]}.items() if value != ''}, axis=1)
            entry = {'时间': str(name), '值': group['temp1'].tolist()}
            time_series_list1.append(entry)
        time_series_list = time_series_list1[::-1]
    elif chartname == '输出':
        df = df[df[VALUE] != 0.0]
        df = df.dropna(subset=[VALUE], how='any')
        grouped = df.groupby(CHARTTIME)
        time_series_list1 = []
        for name, group in grouped:
            group['temp1'] = group.apply(
                lambda row: {key: value for key, value in {'出量': str(row[VALUE]) + str(row[VALUEUOM]),
                                                           '出液名称': row[LABEL]}.items()
                             if value != ''}, axis=1)
            entry = {'时间': str(name), '值': group['temp1'].tolist()}
            time_series_list1.append(entry)
        time_series_list = time_series_list1[::-1]
    elif chartname in ['血常规', '动脉血气分析', '止凝血']:
        time_series_list1 = []
        for group in df:
            group = group.fillna('')
            group['temp'] = group.apply(
                lambda row: {str(row[LABEL]): str(row[VALUE]) + str(row[VALUEUOM])} if str(
                    row[VALUE]) != '' else {},
                axis=1)

            entry = {'时间': str(group.iloc[0][CHARTTIME]), '值': group['temp'].tolist()}
            time_series_list1.append(entry)
        time_series_list = time_series_list1[::-1]
    elif chartname in ['肺部相关', '腺体相关', '眼部相关']:
        time_series_list1 = []
        if not df.empty:
            grouped = df.groupby(CHARTTIME)
            for name, group in grouped:
                group['temp1'] = group.apply(
                    lambda row: {key: value for key, value in {'化验类型': row[SPEC_TYPE_DESC],
                                                               '有机物名称': row[ORG_NAME],
                                                               '分离菌落数目': row[ISOLATE_NUM],
                                                               '抗生素名称': row[AB_NAME],
                                                               '抗生素敏感性': row[DILUTION_TEXT],
                                                               '药物浓度与参考浓度对比': row[DILUTION_COMPARISON],
                                                               '抗生素敏感性时的稀释值': row[DILUTION_VALUE],
                                                               '抗生素的敏感性和试验结果': row[INTERPRETATION]}.items()
                                 if value != ''}, axis=1)
                entry = {'时间': str(name), '值': group['temp1'].tolist(), '病原血部位类别': chartname}
                time_series_list1.append(entry)
        if df_diagnose is not None:
            if not df_diagnose.empty:
                df_diagnose = df_diagnose.drop_duplicates(subset=[LONG_TITLE])
                df_diagnose['temp'] = df_diagnose.apply(
                    lambda row: {'时间': str(illtime), '值': '诊断为-' + str(row[LONG_TITLE]), '病原血部位类别': chartname},
                    axis=1)
                time_series_list1.extend(df_diagnose['temp'].tolist())
        time_series_list = time_series_list1[::-1]
    elif chartname in ['肝脏相关', '心血管系统相关']:
        df['temp'] = df.apply(lambda row: {'时间': str(row[CHARTTIME]),
                                           '值': str(row[LABEL]) + ':' + str(row[VALUE]) + row[VALUEUOM],
                                           '病原血部位类别': chartname},
                              axis=1)
        time_series_list = df['temp'].tolist()
        del df['temp']
    elif chartname == '培养':
        if not df.empty:
            df['temp'] = df.apply(lambda row: {'送检样本': str(row[LABEL]), '时间': str(row[CHARTTIME]),
                                               '值': str(row[VALUE]) + str(
                                                   row[VALUEUOM]).replace('None', '')}, axis=1)
            time_series_list = df['temp'].tolist()
            del df['temp']
        if df_diagnose is not None:
            if not df_diagnose.empty:
                grouped = df_diagnose.groupby(CHARTTIME)
                # print(df_diagnose.columns)
                time_series_list1 = []
                for name, group in grouped:
                    group = group.fillna('')
                    group['temp1'] = group.apply(
                        lambda row: {key: value for key, value in {'有机物名称': row[ORG_NAME],
                                                                   '分离菌落数目': row[ISOLATE_NUM],
                                                                   '抗生素名称': row[AB_NAME],
                                                                   '抗生素敏感性': row[DILUTION_TEXT],
                                                                   '药物浓度与参考浓度对比': row[DILUTION_COMPARISON],
                                                                   '抗生素敏感性时的稀释值': row[DILUTION_VALUE],
                                                                   '抗生素的敏感性和试验结果': row[INTERPRETATION]}.items()
                                     if value != ''}, axis=1)
                    spec_type_desc_values = list(group[SPEC_TYPE_DESC])
                    if len(spec_type_desc_values) > 0:
                        entry = {'时间': str(name), '送检样本': spec_type_desc_values[0], '值': group['temp1'].tolist()}
                        time_series_list1.append(entry)
                time_series_list.extend(time_series_list1[::-1])
    elif chartname == '涂片':
        if not df.empty:
            df['temp'] = df.apply(lambda row: {'时间': str(row[CHARTTIME]),
                                               '值': str(row[LABEL]) + ':' + str(row[VALUE]) + str(
                                                   row[VALUEUOM]).replace('None', '')},
                                  axis=1)
            time_series_list = df['temp'].tolist()
            del df['temp']
        if df_diagnose is not None:
            if not df_diagnose.empty:
                df_diagnose['temp'] = df_diagnose.apply(
                    lambda row: {'时间': str(illtime), '值': '诊断为-' + str(row[LABEL])}, axis=1)
                time_series_list.extend(df_diagnose['temp'].tolist())

    time_series_dict = {chartname: time_series_list}
    return time_series_dict

def get_demographic_info(df_subject_demographic_info, df_charts_illtime_before):
    weight_label_list = ['Daily Weight', 'Admission Weight']
    df_weight = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(weight_label_list), case=False)).any(axis=1)]
    df_weight_kg = df_weight.dropna(subset=[VALUEUOM])
    if len(df_weight_kg) == 0:
        df_weight_kg = df_weight
        selected_rows = df_weight_kg[df_weight_kg[LABEL].str.contains('lbs.')]
        selected_rows[VALUE] = selected_rows[VALUE].astype(float) * 0.4536
        selected_rows[VALUEUOM] = 'kg'
        df_weight_kg.update(selected_rows)
    df_weight_kg = df_weight_kg.drop_duplicates(subset=[CHARTTIME])
    df_weight_kg = df_weight_kg.sort_values(CHARTTIME, ascending=False)
    weight_series_dict = get_time_series(df_weight_kg, '体重')
    if weight_series_dict is not None:
        df_subject_demographic_info.update(weight_series_dict)
    return df_subject_demographic_info

def get_weight(df_charts_illtime_before):
    weight_label_list = ['Daily Weight', 'Admission Weight']
    df_weight = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(weight_label_list), case=False)).any(axis=1)]
    df_weight_kg = df_weight.dropna(subset=[VALUEUOM])
    if len(df_weight_kg) == 0:
        df_weight_kg = df_weight
        selected_rows = df_weight_kg[df_weight_kg[LABEL].str.contains('lbs.')]
        selected_rows[VALUE] = selected_rows[VALUE].astype(float) * 0.4536
        selected_rows[VALUEUOM] = 'kg'
        df_weight_kg.update(selected_rows)
    df_weight_kg = df_weight_kg.drop_duplicates(subset=[CHARTTIME])
    df_weight_kg = df_weight_kg.sort_values(CHARTTIME, ascending=False)
    if len(df_weight_kg) == 0:
        return '体重未知'
    else:
        return df_weight_kg.iloc[0][VALUE]

def get_ill_history(df_charts_illtime_before):
    ill_history_label_list = ['past medical history']
    df_ill_history = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(ill_history_label_list), case=False)).any(axis=1)]
    ill_history = '\n'.join(df_ill_history[VALUE].tolist())
    return ill_history

def get_temperature(df_charts_illtime_before):
    temperature_label_list = ['Temperature Celsius']
    df_temperature = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(temperature_label_list), case=False)).any(axis=1)]
    if df_temperature.empty:
        df_temperature = df_charts_illtime_before[
            df_charts_illtime_before[[LABEL]].apply(
                lambda x: x.str.contains('Temperature Fahrenheit', case=False)).any(axis=1)]
        if not df_temperature.empty:
            df_temperature[VALUE] = df_temperature[VALUE].astype(float)
            df_temperature[VALUE] = ((df_temperature[VALUE] - 32) * 5 / 9).round(2)

    df_temperature = df_temperature[df_temperature[VALUE] != 0]
    df_temperature[VALUEUOM] = '°C'
    tem_time_series_dict = get_time_series(df_temperature, '体温')
    return tem_time_series_dict

def get_blood_pressure(df_charts_illtime_before):
    blood_pressure_label_list = ['Blood Pressure systolic', 'Blood Pressure Diastolic']
    df_blood_pressure = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(blood_pressure_label_list), case=False)).any(
            axis=1)]
    blood_pressure_series_dict = get_time_series(df_blood_pressure, '血压')
    return blood_pressure_series_dict

def get_heart_rate(df_charts_illtime_before):
    heart_rate_label_list = ['Heart Rate']
    df_heart_rate = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(heart_rate_label_list), case=False)).any(axis=1)]
    heart_rate_series_dict = get_time_series(df_heart_rate, '心率')
    return heart_rate_series_dict

def get_respiratory_rate(df_charts_illtime_before):
    respiratory_rate_label_list = ['Respiratory Rate']
    df_respiratory_rate = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(respiratory_rate_label_list), case=False)).any(axis=1)]
    respiratory_rate_series_dict = get_time_series(df_respiratory_rate, '呼吸频率')
    return respiratory_rate_series_dict

def get_consciousness(df_charts_illtime_before):

    consciousness_label_list = ['GcsApacheIIScore','GcsScore_ApacheIV']
    df_consciousness = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(consciousness_label_list), case=False)).any(axis=1)]
    if len(df_consciousness) > 0:
        df_consciousness[VALUE] = pd.to_numeric(df_consciousness[VALUE], errors='coerce')
        df_consciousness[VALUE] = np.where(df_consciousness[VALUE] <= 14, '意识改变', '意识正常')
        df_consciousness = df_consciousness[
            df_consciousness[[VALUE]].apply(
                lambda x: x.str.contains('意识', case=False)).any(axis=1)]
    else:
        gcs_label = ['GCS - Eye Opening', 'GCS - Verbal Response', 'GCS - Motor Response']
        df_consciousness = df_charts_illtime_before[df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(gcs_label), case=False)).any(axis=1)]

        if len(df_consciousness) > 0 and (len(df_consciousness) % 3 == 0):
            df_consciousness = get_gcs(df_consciousness)
            df_consciousness[VALUE] = pd.to_numeric(df_consciousness[VALUE], errors='coerce')
            df_consciousness[VALUE] = np.where(df_consciousness[VALUE] <= 14, '意识改变', '意识正常')
            df_consciousness = df_consciousness[
                df_consciousness[[VALUE]].apply(
                    lambda x: x.str.contains('意识', case=False)).any(axis=1)]
    consciousness_series_dict = get_time_series(df_consciousness, '意识')
    return consciousness_series_dict

def get_gcs(df_gcs):
    grouped_df = df_gcs.groupby([CHARTTIME])
    for group_key, group_index in grouped_df.groups.items():
        group = grouped_df.get_group(group_key)
        count = sum(group[VALUENUM])
        group_index_list = group_index.tolist()
        df_gcs.loc[group_index_list, VALUE] = count
    df_gcs = df_gcs.drop_duplicates(subset=[SUBJECT_ID, CHARTTIME, VALUE])
    df_gcs = df_gcs.sort_values(CHARTTIME, ascending=False)
    return df_gcs

def get_blood_pressure_systolic(df_charts_illtime_before):
    blood_pressure_label_list = ['Blood Pressure systolic']
    df_blood_pressure = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(blood_pressure_label_list), case=False)).any(
            axis=1)]
    blood_pressure_series_dict = get_time_series(df_blood_pressure, '收缩压')
    return blood_pressure_series_dict

def get_input(df_input):
    df_input = df_input.drop_duplicates()
    df_input = df_input.dropna(subset=[AMOUNT], how='any')
    df_input = df_input.fillna('')
    df_input = df_input[df_input[AMOUNT] != 0]
    input_series_dict = get_time_series(df_input, '输入')
    return input_series_dict

def get_output(df_output_illtime_before):
    output_series_dict = get_time_series(df_output_illtime_before, '输出')
    return output_series_dict

def get_blood_routine(df_charts_illtime_before):
    blood_routine_label_list = ['WBC', 'WBC Count', 'RBC', ' Rbc', 'RBC, Ascites', 'RBC, CSF', 'RBC, Joint Fluid', 'RBC, Pleural',
    'Hemoglobin', 'Absolute Hemoglobin', 'Hematocrit', 'Hematocrit, Ascites', 'Hematocrit, CSF',
    'Hematocrit, Joint Fluid', 'Hematocrit, Pleural', 'MCV', 'MCH', 'MCHC', 'Platelet Count', 'Platelets',
    'Atypical Lymphocytes', 'Lymphocytes, Percent', 'Lymphocytes', 'Monocytes', 'Neutrophils', 'Eosinophils',
    'Basophils', 'Absolute Lymphocyte Count', 'Nucleated Red Cells', 'Nucleated RBC', 'Large Platelets',
    'C-Reactive Protein']
    blood_routine_dict = {
        'WBC': '白细胞计数',
        'WBC Count': '白细胞计数',
        'RBC': '红细胞计数',
        ' Rbc': '红细胞计数',
        'RBC, Ascites': '红细胞计数',
        'RBC, CSF': '红细胞-脑脊液',
        'RBC, Joint Fluid': '红细胞-关节液',
        'RBC, Pleural': '红细胞-胸膜',
        'Hemoglobin': '血红蛋白浓度',
        'Absolute Hemoglobin':'血红蛋白浓度',
        'Hematocrit': '红细胞压积',
        'Hematocrit, Ascites': '红细胞压积',
        'Hematocrit, CSF': '红细胞压积-脑脊液',
        'Hematocrit, Joint Fluid': '红细胞压积-关节液',
        'Hematocrit, Pleural': '红细胞压积-胸膜',
        'MCV': '红细胞平均体积 (MCV)',
        'MCH': '平均血红蛋白量 (MCH)',
        'MCHC': '平均血红蛋白浓度 (MCHC)',
        'Platelet Count': '血小板计数',
        'Platelets': '血小板计数',
        'Atypical Lymphocytes':'非典型淋巴细胞',
        'Lymphocytes, Percent': '淋巴细胞百分比',
        'Lymphocytes': '淋巴细胞',
        'Monocytes': '单核细胞比值',
        'Neutrophils': '中性粒细胞比值',
        'Eosinophils': '嗜酸性粒细胞比值',
        'Basophils': '嗜碱性粒细胞比值',
        'Absolute Lymphocyte Count': '淋巴细胞绝对值',
        'Nucleated Red Cells': '有核红细胞比值',
        'Nucleated RBC': '有核红细胞比值',
        'Large Platelets': '大型血小板比率',
        'C-Reactive Protein': 'C反应蛋白',
    }
    df_blood_routine = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(blood_routine_label_list), case=False)).any(
            axis=1)]
    df_blood_routine[LABEL] = df_blood_routine[LABEL].map(blood_routine_dict)
    df_blood_routine = df_blood_routine.dropna(subset=[LABEL], how='any')
    df_blood_routine = df_blood_routine.drop_duplicates(subset=[SUBJECT_ID, CHARTTIME, LABEL])
    grouped_dfs = get_batches(df_blood_routine)
    blood_routine_series_dict = get_time_series(grouped_dfs, '血常规')
    return blood_routine_series_dict

def get_pathogenic_blood(df_labevents_illtime_before, df_microbiology_illtime_before, df_diagnose, illtime):
    df_microbiology_illtime_before = df_microbiology_illtime_before.dropna(subset=[ORG_NAME])
    # pd.set_option('display.max_columns', None)
    combined_list = []
    pathogenic_blood_fei = {
        'LEGIONELLA PNEUMOPHILA ': '嗜肺军团菌血清1型AbIgM',
        'Pneu mycplsm pneumoniae': '肺炎支原体AbIgM',
        'Pneumonia d/t chlamydia': '肺炎衣原体AbIgM',
        'RESPIRATORY SYNCYTIAL VIRUS (RSV)': '呼吸道合胞病毒AbIgM',
        'Respiratory Syncytial Virus': '呼吸道合胞体病毒',
        'Acute Bronchiolitis due to RSV': '由RSV引起的急性支气管炎',
        'Respiratory Syncytial Viral Pneumonia': 'RSV引起的呼吸道合胞体病毒性肺炎',
        'Vaccination for RSV': 'RSV疫苗接种',
        'Influenza A/B by DFA': '通过DFA检测的甲/乙型流感病毒',
        'Influenza A Virus': '甲型流感病毒',
        'Influenza B Virus': '乙型流感病毒',
        'Parainfluenza Virus Type 1': '副流感病毒1型',
        'Parainfluenza Virus Type 3': '副流感病毒3型',
        'Positive for Influenza A Viral Antigen': '检测到甲型流感病毒抗原阳性',
        'Positive for Influenza B Viral Antigen': '检测到乙型流感病毒抗原阳性',
        'Parainfluenza Viral Pneumonia': '副流感病毒性肺炎',
        'Influenza with Pneumonia': '流感并发肺炎'
    }
    pathogenic_blood_fei_list = list(pathogenic_blood_fei.keys())

    # fei_mask = df_diagnose[LONG_TITLE].apply(
    #     lambda x: any(substring.lower() in x.lower() for substring in pathogenic_blood_fei_list)) | \
    #     df_diagnose[LONG_TITLE].apply(
    #                lambda x: any(substring.lower() in x.lower() for substring in pathogenic_blood_fei_list))

    fei_mask = df_diagnose[LONG_TITLE].str.contains('|'.join(pathogenic_blood_fei_list), case=False) | \
               df_diagnose[LONG_TITLE].str.contains('|'.join(pathogenic_blood_fei_list), case=False)
    df_fei_diagnose = df_diagnose[fei_mask]
    df_fei_microbio_diagnose = df_microbiology_illtime_before[
        df_microbiology_illtime_before[ORG_NAME].str.contains('|'.join(pathogenic_blood_fei_list), case=False)]
    # print(df_fei_diagnose)
    # print(df_fei_microbio_diagnose)

    df_fei_microbio_diagnose[ORG_NAME] = df_fei_microbio_diagnose[ORG_NAME].map(pathogenic_blood_fei)
    fei_series_dict = get_time_series(df_fei_microbio_diagnose, '肺部相关', df_diagnose=df_fei_diagnose, illtime=illtime)
    if fei_series_dict is not None:
        fei_series_list = fei_series_dict['肺部相关']
        # print(fei_series_list)
        combined_list = combined_list + fei_series_list

        # combined_list.extend(fei_series_list)

    pathogenic_blood_xian = {
        'ADENOVIRUS': '腺病毒AbIgM',
        'Intes infec adenovirus': '肠道感染腺病毒',
        'Adenoviral meningitis': '腺病毒性脑膜炎',
        'Adenovirus infect NOS': '腺病毒感染，未明确部位',
        'Adenoviral pneumonia': '腺病毒性肺炎'
    }
    pathogenic_blood_xian_list = list(pathogenic_blood_xian.keys())
    mask = df_diagnose[LONG_TITLE].str.contains('|'.join(pathogenic_blood_xian_list), case=False) | \
           df_diagnose[LONG_TITLE].str.contains('|'.join(pathogenic_blood_xian_list), case=False)
    df_xian_diagnose = df_diagnose[mask]
    df_xian_microbio_diagnose = df_microbiology_illtime_before[
        df_microbiology_illtime_before[ORG_NAME].str.contains('|'.join(pathogenic_blood_xian_list), case=False)]
    # print(df_xian_diagnose)
    # print(df_xian_microbio_diagnose)
    df_xian_microbio_diagnose[ORG_NAME] = df_xian_microbio_diagnose[ORG_NAME].map(pathogenic_blood_xian)
    xian_series_dict = get_time_series(df_xian_microbio_diagnose, '腺体相关', df_diagnose=df_xian_diagnose, illtime=illtime)
    if xian_series_dict is not None:
        xian_series_list = xian_series_dict['腺体相关']
        combined_list.extend(xian_series_list)

    pathogenic_blood_eye = {
        'CHLAMYDIA TRACHOMATIS': '沙眼衣原体'
    }
    pathogenic_blood_eye_list = list(pathogenic_blood_eye.keys())
    df_eye_microbio_diagnose = df_microbiology_illtime_before[
        df_microbiology_illtime_before[ORG_NAME].str.contains('|'.join(pathogenic_blood_eye_list), case=False)]
    # print(df_eye_microbio_diagnose)
    df_eye_microbio_diagnose[ORG_NAME] = df_eye_microbio_diagnose[ORG_NAME].map(pathogenic_blood_eye)
    eye_series_dict = get_time_series(df_eye_microbio_diagnose, '眼部相关')
    if eye_series_dict is not None:
        eye_series_list = eye_series_dict['眼部相关']
        combined_list.extend(eye_series_list)

    pathogenic_blood_gan = {
        'Hepatitis A Virus IgM Antibody': '乙肝病毒IgM抗体',
        'Hepatitis B Core Antibody, IgM': '乙肝核心抗体IgM'
    }
    pathogenic_blood_gan_list = list(pathogenic_blood_gan.keys())
    df_gan_microbio_diagnose = df_labevents_illtime_before[
        df_labevents_illtime_before[LABEL].str.contains('|'.join(pathogenic_blood_gan_list), case=False)]
    # print(df_gan_microbio_diagnose)
    df_gan_microbio_diagnose[LABEL] = df_gan_microbio_diagnose[LABEL].map(pathogenic_blood_gan)
    gan_series_dict = get_time_series(df_gan_microbio_diagnose, '肝脏相关')
    if gan_series_dict is not None:
        gan_series_list = gan_series_dict['肝脏相关']
        combined_list.extend(gan_series_list)

    pathogenic_blood_xin = {
        'Anticardiolipin Antibody IgM': '抗心磷脂抗体IgM'
    }
    pathogenic_blood_xin_list = list(pathogenic_blood_xin.keys())
    df_xin_microbio_diagnose = df_labevents_illtime_before[
        df_labevents_illtime_before[LABEL].str.contains('|'.join(pathogenic_blood_xin_list), case=False)]
    # print(df_xin_microbio_diagnose)
    df_xin_microbio_diagnose[LABEL] = df_xin_microbio_diagnose[LABEL].map(pathogenic_blood_xin)
    xin_series_dict = get_time_series(df_xin_microbio_diagnose, '心血管系统相关')
    if xin_series_dict is not None:
        xin_series_list = xin_series_dict['心血管系统相关']
        combined_list.extend(xin_series_list)
    # print(combined_list)
    pathogenic_blood_dict = {'病原血检查': combined_list}
    if len(combined_list) == 0:
        return None
    # print(pathogenic_blood_dict)
    return pathogenic_blood_dict

# def get_radio(df_radio_illtime_before):
#     # radio_series_dict = get_time_series(df_radio_illtime_before, '影像报告')
#     return radio_series_dict

def get_gas_analysis(df_charts_illtime_before):
    blood_gas_analysis_label_list = ["Albumin", "Alveolar-arterial Gradient", "Base Excess", "Calculated Bicarbonate", "Calculated Total CO2",
    "Carboxyhemoglobin", "Chloride", "Creatinine", "Estimated GFR (MDRD equation)", "Free Calcium", "Glucose",
    "Hematocrit", "Hematocrit, Calculated", "Hemoglobin", "% Ionized Calcium", "Lactate", "Lithium", "Methemoglobin",
    "O2 Flow", "Osmolality", "Oxygen", "Oxygen Saturation", "P50 of Hemoglobin", "pCO2", "PEEP", "pH", "pH, Urine",
    "pO2, Blood", "pO2, Fluid", "pO2, Body Fluid", "Potassium", "Required O2", "Sodium, Body Fluid", "Sodium, Urine",
    "Sodium, Whole Blood", "Temperature", "Total Calcium", "WB tCO2", "HCO3ApacheIIValue", "HCO3 (serum)",
    "Inspired O2 Fraction", "FiO2ApacheIIValue"]
    gas_analysis_dict = {
    "Albumin": "白蛋白",
    "Alveolar-arterial Gradient": "肺泡-动脉梯度",
    "Base Excess": "碱剩余",
    "Calculated Bicarbonate": "碳酸氢盐",
    "Calculated Total CO2": "总二氧化碳",
    "Carboxyhemoglobin": "碳氧血红蛋白",
    "Chloride": "氯离子",
    "Creatinine": "肌酐",
    "Estimated GFR (MDRD equation)": "肾小球滤过率",
    "Free Calcium": "游离钙",
    "Glucose": "葡萄糖",
    "Hematocrit": "红细胞压积",
    "Hematocrit, Calculated": "红细胞压积",
    "Hemoglobin": "血红蛋白",
    "% Ionized Calcium": "游离钙百分比",
    "Lactate": "乳酸",
    "Lithium": "锂",
    "Methemoglobin": "高铁血红蛋白",
    "O2 Flow": "氧气流量",
    "Osmolality": "渗透压",
    "Oxygen": "氧气",
    "Oxygen Saturation": "血氧饱和度",
    "P50 of Hemoglobin": "血红蛋白的P50值",
    "pCO2": "二氧化碳分压",
    "PEEP": "呼气末正压",
    "pH": "pH值",
    "pH, Urine": "尿液pH值",
    "pO2, Blood": "血液氧分压",
    "pO2, Fluid": "体液氧分压",
    "pO2, Body Fluid": "体液氧分压",
    "Potassium": "钾",
    "Required O2": "所需氧气",
    "Sodium, Body Fluid": "体液中的钠",
    "Sodium, Urine": "尿液中的钠",
    "Sodium, Whole Blood": "全血中的钠",
    "Temperature": "温度",
    "Total Calcium": "总钙",
        "WB tCO2": "全血总二氧化碳",
        'HCO3ApacheIIValue': '实际碳酸氢根',
        'HCO3 (serum)': '碳酸氢根(血清)',
        'Inspired O2 Fraction': '吸入氧浓度',
        'FiO2ApacheIIValue': '吸入氧浓度'
}
    df_blood_gas_analysis = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(blood_gas_analysis_label_list), case=False)).any(
            axis=1)]
    df_blood_gas_analysis[LABEL] = df_blood_gas_analysis[LABEL].map(gas_analysis_dict)
    df_blood_gas_analysis = df_blood_gas_analysis.dropna(subset=[LABEL], how='any')
    df_blood_gas_analysis = df_blood_gas_analysis.drop_duplicates(
        subset=[SUBJECT_ID, CHARTTIME, LABEL])
    grouped_gas_analysis = get_batches(df_blood_gas_analysis)
    gas_analysis_series_dict = get_time_series(grouped_gas_analysis, '动脉血气分析')
    return gas_analysis_series_dict

def get_hemostasis(df_charts_illtime_before):
    hemostasis_label_list = ['PT', 'INR', 'Activated Clotting Time', 'Fibrinogen', 'Fibrinogen', 'D-Dimer']
    hemostasis_dict = {
         'PT': '凝血酶原时间',
        'INR(PT)': '国际标准化比值',
        'INR': '国际标准化比值',
        'Fibrinogen': '纤维蛋白原',
        'Activated Clotting Time': '部分凝血酶原时间',
        'D-Dimer': 'D_二聚体（II）',
        'D-Dimer (SOFT)': 'D_二聚体（II）'
    }

    df_hemostasis = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(hemostasis_label_list), case=False)).any(axis=1)]
    df_hemostasis[LABEL] = df_hemostasis[LABEL].map(hemostasis_dict)
    df_hemostasis = df_hemostasis.dropna(subset=[LABEL], how='any')
    df_hemostasis = df_hemostasis.drop_duplicates(
        subset=[SUBJECT_ID, CHARTTIME, LABEL])
    grouped_hemostasis = get_batches(df_hemostasis)
    hemostasis_series_dict = get_time_series(grouped_hemostasis, '止凝血')
    return hemostasis_series_dict

def get_culture(df_charts_illtime_before, df_microbiology_illtime_before, df_procedure_illtime_before):
    df_microbiology_illtime_before = df_microbiology_illtime_before.dropna(subset=[ORG_NAME])
    culture_dict = {
        'Blood Cultured': '血培养',
        'Arterial Line Tip Cultured': '动脉导管尖端培养',
        'Sheath Line Tip Cultured': '管路尖端培养',
        'FLUID RECEIVED IN BLOOD CULTURE BOTTLES': '收到的血培养瓶中的液体',
        'Stool Culture': '大便培养',
        'Pan Culture': '全面培养',
        'Urine culture': '尿液培养',
        'Sputum Culture': '痰液培养',
        'PA Catheter Line Tip Cultured': 'PA导管尖端培养',
        'Trauma Line Tip Cultured': '创伤导管尖端培养',
        'POST-MORTEM VIRAL CULTURE': '尸检病毒培养',
        'CSF Culture': '脑脊液培养',
        'Triple Introducer Line Tip Cultured': '三腔导管尖端培养',
        'Multi Lumen Line Tip Cultured': '多腔导管尖端培养',
        'Rapid Respiratory Viral Screen & Culture': '快速呼吸道病毒筛查和培养',
        'surface cultures': '表面培养',
        'PICC Line Tip Cultured': 'PICC导管尖端培养',
        'ANORECTAL/VAGINAL CULTURE': '肛门/阴道培养',
        'THROAT CULTURE': '咽喉培养',
        'VIRAL CULTURE:R/O HERPES SIMPLEX VIRUS': '病毒培养：排除单纯疱疹病毒',
        'Dialysis Catheter Tip Cultured': '透析导管尖端培养',
        'ICP Line Tip Cultured': 'ICP导管尖端培养',
        'BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)': '血培养（真菌/溶血瓶）',
        'Tunneled (Hickman) Line Tip Cultured': '隧道式（Hickman）导管尖端培养',
        'Blood Culture Hold': '血培养等待',
        'Stem Cell - Blood Culture': '干细胞 - 血培养',
        'POSTMORTEM CULTURE': '尸检培养',
        'Pheresis Catheter Line Tip Cultured': '分离导管尖端培养',
        'FOOT CULTURE': '足部培养',
        'sputum culture': '痰液培养',
        'urine culture': '尿液培养',
        "Midline Tip Cultured": "中心静脉尖端培养",
        "AVA Line Tip Cultured": "动脉血管通道尖端培养",
        "BLOOD CULTURE (POST-MORTEM)": "血液培养（尸检）",
        "STERILITY CULTURE": "无菌培养",
        "Blood Cultures": "血液培养",
        "Wound Culture": "伤口培养",
        "Cordis/Introducer Line Tip Cultured": "Cordis/导管插入处尖端培养",
        "Urine Culture": "尿培养",
        "VIRAL CULTURE: R/O CYTOMEGALOVIRUS": "病毒培养：排除巨细胞病毒",
        "BAL Fluid Culture": "肺泡灌洗液培养",
        "Presep Catheter Line Tip Cultured": "Presep导管尖端培养",
        "BLOOD CULTURE - NEONATE": "新生儿血培养",
        "CCO PAC Line Tip Cultured": "CCO PAC导管尖端培养",
        "VARICELLA-ZOSTER CULTURE": "水痘-带状疱疹病毒培养",
        'Urinalysis sent': '尿液样本送检'
    }
    df_charts_illtime_before = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(lambda x: x.str.contains('culture', case=False)).any(axis=1)]
    df_charts_illtime_before = df_charts_illtime_before[df_charts_illtime_before[VALUE] != '']
    df_procedure_illtime_before = df_procedure_illtime_before[
        df_procedure_illtime_before[[LABEL]].apply(lambda x: x.str.contains('culture', case=False)).any(axis=1)]
    df_procedure_illtime_before = df_procedure_illtime_before[df_procedure_illtime_before[VALUE] != '']
    df_cluture = pd.concat([df_charts_illtime_before, df_procedure_illtime_before])
    df_cluture[LABEL] = df_cluture[LABEL].map(culture_dict)
    df_microbiology_illtime_before = df_microbiology_illtime_before[
        df_microbiology_illtime_before[[SPEC_TYPE_DESC]].apply(lambda x: x.str.contains('culture', case=False)).any(
            axis=1)]
    df_microbiology_illtime_before = df_microbiology_illtime_before[df_microbiology_illtime_before[ORG_NAME] != '']
    df_microbiology_illtime_before[SPEC_TYPE_DESC] = df_microbiology_illtime_before[SPEC_TYPE_DESC].map(
        culture_dict)
    pd.set_option('display.max_columns', None)
    culture_series = get_time_series(df_cluture, '培养', df_diagnose=df_microbiology_illtime_before)
    return culture_series

def get_smear(df_charts_illtime_before, df_diagnose, illtime):
    df_charts_illtime_before = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(lambda x: x.str.contains('smear', case=False)).any(axis=1)]
    df_charts_illtime_before = df_charts_illtime_before[df_charts_illtime_before[VALUE] != '']
    df_diagnose = df_diagnose[
        df_diagnose[[LONG_TITLE]].apply(lambda x: x.str.contains('smear', case=False)).any(axis=1)]
    df_diagnose.rename(columns={LONG_TITLE: LABEL}, inplace=True)
    smear_dict = {
        'Inpatient Hematology/Oncology Smear': '住院血液学/肿瘤学涂片',
        'Blood Parasite Smear': '血液寄生虫涂片',
        'Abn pap cervix HPV NEC': '异常宫颈HPV非特指型Pap涂片',
        'Abn gland pap smr vagina': '异常腺体阴道Pap涂片',
        'H/O Smear': 'H/O涂片',
        'Platelet Smear': '血小板涂片',
        'Pap smear anus w ASC-US': '带有ASC-US的肛门Pap涂片',
        'Abn glandular pap smear': '异常腺体Pap涂片'
    }
    df_charts_illtime_before = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(lambda x: x.str.contains('smear', case=False)).any(axis=1)]
    df_diagnose = df_diagnose[df_diagnose[[LABEL]].apply(lambda x: x.str.contains('smear', case=False)).any(axis=1)]
    df_charts_illtime_before[LABEL] = df_charts_illtime_before[LABEL].map(smear_dict)
    df_diagnose[LABEL] = df_diagnose[LABEL].map(smear_dict)
    smear_series_dict = get_time_series(df_charts_illtime_before, '涂片', df_diagnose=df_diagnose, illtime=illtime)
    return smear_series_dict

def get_qsofa(current_list, start_time, end_time, supply_base_data_flag, respiratory_rate_dict, consciousness_dict,
              blood_pressure_systolic_dict):
    is_have_qsofa = False
    qsofa_sc = False
    if respiratory_rate_dict is not None and consciousness_dict is not None and blood_pressure_systolic_dict is not None:
        respiratory_rate = [temp for temp in respiratory_rate_dict['呼吸频率'] if
                            start_time <= pd.to_datetime(temp['时间']) < end_time]
        consciousness = [temp for temp in consciousness_dict['意识'] if
                         start_time <= pd.to_datetime(temp['时间']) < end_time]
        blood_pressure_systolic = [temp for temp in blood_pressure_systolic_dict['收缩压'] if
                                   start_time <= pd.to_datetime(temp['时间']) < end_time]

        if len(respiratory_rate) == 0:
            respiratory_rate = [temp for temp in respiratory_rate_dict['呼吸频率'] if
                                start_time - timedelta(hours=1.5) <= pd.to_datetime(temp['时间']) <= end_time + timedelta(
                                    hours=1.5)]
            supply_base_data_flag.append('呼吸频率')
        respiratory_rate_current = None
        if len(respiratory_rate) > 0:
            respiratory_rate = max(respiratory_rate, key=lambda x: datetime.strptime(x['时间'], '%Y-%m-%d %H:%M:%S'))
            respiratory_rate_current = respiratory_rate['值']

        if len(consciousness) == 0:
            consciousness = [temp for temp in consciousness_dict['意识'] if
                             start_time - timedelta(hours=5) <= pd.to_datetime(temp['时间']) <= end_time + timedelta(
                                 hours=5)]
            supply_base_data_flag.append('意识')
        consciousness_current = None
        if len(consciousness) > 0:
            consciousness = max(consciousness, key=lambda x: datetime.strptime(x['时间'], '%Y-%m-%d %H:%M:%S'))
            consciousness_current = consciousness['值']

        if len(blood_pressure_systolic) == 0:
            blood_pressure_systolic = [temp for temp in blood_pressure_systolic_dict['收缩压'] if
                                       start_time - timedelta(hours=1.5) <= pd.to_datetime(
                                           temp['时间']) <= end_time + timedelta(
                                           hours=1.5)]
            supply_base_data_flag.append('收缩压')
        blood_pressure_systolic_current = None
        if len(blood_pressure_systolic) > 0:
            blood_pressure_systolic = max(blood_pressure_systolic,
                                          key=lambda x: datetime.strptime(x['时间'], '%Y-%m-%d %H:%M:%S'))
            blood_pressure_systolic_current = blood_pressure_systolic['值']

        qsofa_score = 0
        if respiratory_rate_current is not None and (int(float(re.findall(r'\d+', respiratory_rate_current)[0])) >= 22):
            qsofa_score += 1
        if blood_pressure_systolic_current is not None and (
                int(float(re.findall(r'\d+', blood_pressure_systolic_current)[0])) <= 100):
            qsofa_score += 1
        if consciousness_current is not None and consciousness_current == '意识改变':
            qsofa_score += 1
        if respiratory_rate_current is not None and (blood_pressure_systolic_current is not None) and (
                consciousness_current is not None):
            qsofa = {'QSOFA': {'时间': str(start_time), '呼吸频率': respiratory_rate_current,
                               '收缩压': blood_pressure_systolic_current,
                               '意识': consciousness_current, 'QSOFA分数': qsofa_score}}

            current_list.append(qsofa)
            is_have_qsofa = True
            qsofa_sc = qsofa_score
        qsofa_new = []
        respiratory_rate_new = [temp for temp in respiratory_rate_dict['呼吸频率'] if
                                pd.to_datetime(temp['时间']) < start_time]
        consciousness_new = [temp for temp in consciousness_dict['意识'] if pd.to_datetime(temp['时间']) < start_time]
        blood_pressure_systolic_new = [temp for temp in blood_pressure_systolic_dict['收缩压'] if
                                       pd.to_datetime(temp['时间']) < start_time]
        if len(respiratory_rate_new) != 0:
            if len(respiratory_rate_new) > 24:
                respiratory_rate_new = respiratory_rate_new[:24]
            qsofa_new.append({'呼吸频率': respiratory_rate_new})
        if len(consciousness_new) != 0:
            if len(consciousness_new) > 0:
                consciousness_new = consciousness_new[:24]
            qsofa_new.append({'意识': consciousness_new})
        if len(blood_pressure_systolic_new) != 0:
            if len(blood_pressure_systolic_new) > 0:
                blood_pressure_systolic_new = blood_pressure_systolic_new[:24]
            qsofa_new.append({'收缩压': blood_pressure_systolic_new})
        if len(qsofa_new) == 0:
            qsofa_new = None
        return current_list, qsofa_new, is_have_qsofa, qsofa_sc
    return current_list, None, is_have_qsofa, qsofa_sc

def get_base_current(subject_id, hadm_id, start_time, end_time, temperature_dict, blood_pressure_dict,
                     heart_rate_dict,
                     input_dict, output_dict, respiratory_rate_dict, consciousness_dict, blood_pressure_systolic_dict):
    current_list = []
    supply_base_data_flag = []
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    flag = is_have_adverse_events(hadm_id, end_time)
    current_list, temperature_dict_new, supply_base_data_flag = getcurrent(supply_base_data_flag, flag, start_time,
                                                                           start_time - timedelta(hours=4), end_time,
                                                                           end_time + timedelta(hours=4),
                                                                           temperature_dict, '体温', current_list)
    current_list, blood_pressure_dict_new, supply_base_data_flag = getcurrent(supply_base_data_flag, flag, start_time,
                                                                              start_time - timedelta(hours=1.5),
                                                                              end_time,
                                                                              end_time + timedelta(hours=1.5),
                                                                              blood_pressure_dict, '血压', current_list)
    current_list, heart_rate_dict_new, supply_base_data_flag = getcurrent(supply_base_data_flag, flag, start_time,
                                                                          start_time - timedelta(hours=1.5), end_time,
                                                                          end_time + timedelta(hours=1.5),
                                                                          heart_rate_dict, '心率', current_list)
    # 输入输出的历史包含当前信息，所以需要看endtime之前
    current_list, input_dict_new, supply_base_data_flag = getcurrent(supply_base_data_flag, flag, end_time,
                                                                     start_time - timedelta(hours=3), end_time,
                                                                     end_time + timedelta(hours=3),
                                                                     input_dict, '输入', current_list)
    current_list, output_dict_new, supply_base_data_flag = getcurrent(supply_base_data_flag, flag, end_time,
                                                                      start_time - timedelta(hours=3), end_time,
                                                                      end_time + timedelta(hours=3),
                                                                      output_dict, '输出', current_list)
    input_sum = 0
    output_sum = 0
    # print(input_dict)
    if input_dict is not None:
        for temp_time in input_dict['输入']:
            if (pd.to_datetime(temp_time['时间']) < end_time) and (
                    pd.to_datetime(temp_time['时间']) >= end_time - timedelta(hours=24)):
                for temp in temp_time['值']:
                    value_unit = str(temp['入量'])
                    if re.search(r'ml|mg', value_unit):
                        numbers = re.sub(r'[^\d.]', '', value_unit)
                        # print(numbers)
                        input_sum = input_sum + float(numbers)
    if output_dict is not None:
        for temp_time in output_dict['输出']:
            if (pd.to_datetime(temp_time['时间']) < end_time) and (
                    pd.to_datetime(temp_time['时间']) >= end_time - timedelta(hours=24)):
                for temp in temp_time['值']:
                    value_unit = str(temp['出量'])
                    if re.search(r'ml|mg', value_unit):
                        numbers = re.sub(r'[^\d.]', '', value_unit)
                        # print(numbers)
                        output_sum = output_sum + float(numbers)
    body_fluid_dict = {'体液24h': str(round(input_sum, 2)) + '/' + str(round(output_sum, 2)) + 'ml'}
    current_list.append(body_fluid_dict)

    current_list, qsofa_new, ishave_qsofa, qsofa_sc = get_qsofa(current_list, start_time, end_time,
                                                                supply_base_data_flag, respiratory_rate_dict,
                                                                consciousness_dict, blood_pressure_systolic_dict)
    if len(current_list) == 0:
        current_list = None
    if len(supply_base_data_flag) == 0:
        supply_base_data_flag = None
    return current_list, temperature_dict_new, blood_pressure_dict_new, heart_rate_dict_new, input_dict_new, output_dict_new, qsofa_new, supply_base_data_flag, ishave_qsofa, qsofa_sc

def getcurrent(supply_data_flag, flag, start_time, start_time_not_ae, end_time, end_time_not_ae, chart_dict, str_kind,
               current_list):
    if chart_dict is not None and str(chart_dict) != 'None':
        if isinstance(chart_dict, dict):
            # print('========================')
            # print(chart_dict)
            chart_after = [temp for temp in chart_dict[str_kind] if pd.to_datetime(temp['时间']) < start_time]
            if len(chart_after) == 0:
                dict_after = None
            else:
                dict_after = {str_kind: chart_after}
            if str_kind == '输入' or str_kind == '输出':
                return current_list, dict_after, supply_data_flag

            chart = [temp for temp in chart_dict[str_kind] if
                     start_time <= pd.to_datetime(temp['时间']) < end_time]
            if len(chart) > 0:
                # chart = max(chart, key=lambda x: datetime.strptime(x['时间'], '%Y-%m-%d %H:%M:%S'))
                chart = min(chart, key=lambda x: pd.to_datetime(x['时间']))

                chart['时间'] = str(start_time)
                current = {str_kind: chart}
                current_list.append(current)
                return current_list, dict_after, supply_data_flag
            else:
                if flag:
                    return current_list, dict_after, supply_data_flag
                else:
                    # chart = [temp for temp in dict[str_kind] if
                    #          (start_time_not_ae) <= datetime.strptime(temp['时间'], '%Y-%m-%d %H:%M:%S') <= end_time_not_ae]
                    chart = [temp for temp in chart_dict[str_kind] if
                             (start_time_not_ae) <= pd.to_datetime(temp['时间']) <= end_time_not_ae]
                    if len(chart) > 0:
                        mid_time = start_time + (end_time - start_time) / 2
                        # chart = min(chart, key=lambda x: abs(
                        #     datetime.strptime(x['时间'], '%Y-%m-%d %H:%M:%S') - mid_time))
                        chart = min(chart, key=lambda x: abs(pd.to_datetime(x['时间']) - mid_time))
                        chart['时间'] = str(start_time)
                        current = {str_kind: chart}
                        current_list.append(current)
                        supply_data_flag.append(str_kind)
                        return current_list, dict_after, supply_data_flag
                    else:
                        return current_list, dict_after, supply_data_flag
    return current_list, chart_dict, supply_data_flag

def get_next_current(subject_id, hadm_id, start_time, end_time, blood_routine_dict, pathogenic_blood_dict, radio_dict,
                     gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict):
    next_current_list = []
    supply_next_data_flag = []
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    flag = is_have_adverse_events(hadm_id, end_time)
    next_current_list, blood_routine_dict, supply_next_data_flag = getcurrent(supply_next_data_flag, flag, start_time,
                                                                              start_time - timedelta(hours=12),
                                                                              end_time,
                                                                              end_time + timedelta(hours=12),
                                                                              blood_routine_dict, '血常规',
                                                                              next_current_list)
    next_current_list, pathogenic_blood_dict, supply_next_data_flag = getcurrent(supply_next_data_flag, flag,
                                                                                 start_time,
                                                                                 start_time - timedelta(hours=12),
                                                                                 end_time,
                                                                                 end_time + timedelta(hours=12),
                                                                                 pathogenic_blood_dict,
                                                                                 '病原血检查', next_current_list)
    next_current_list, radio_dict, supply_next_data_flag = getcurrent(supply_next_data_flag, flag, start_time,
                                                                      start_time - timedelta(hours=24), end_time,
                                                                      end_time + timedelta(hours=24),
                                                                      radio_dict, '影像报告', next_current_list)
    next_current_list, gas_analysis_dict, supply_next_data_flag = getcurrent(supply_next_data_flag, flag, start_time,
                                                                             start_time - timedelta(hours=12), end_time,
                                                                             end_time + timedelta(hours=12),
                                                                             gas_analysis_dict, '动脉血气分析',
                                                                             next_current_list)
    next_current_list, hemostasis_dict, supply_next_data_flag = getcurrent(supply_next_data_flag, flag, start_time,
                                                                           start_time - timedelta(hours=12), end_time,
                                                                           end_time + timedelta(hours=12),
                                                                           hemostasis_dict, '止凝血', next_current_list)
    next_current_list, culture_dict, supply_next_data_flag = getcurrent(supply_next_data_flag, flag, start_time,
                                                                        start_time - timedelta(hours=12), end_time,
                                                                        end_time + timedelta(hours=12),
                                                                        culture_dict, '培养', next_current_list)
    next_current_list, smear_dict, supply_next_data_flag = getcurrent(supply_next_data_flag, flag, start_time,
                                                                      start_time - timedelta(hours=12), end_time,
                                                                      end_time + timedelta(hours=12),
                                                                      smear_dict, '涂片', next_current_list)

    if len(next_current_list) == 0:
        next_current_list = None
    if len(supply_next_data_flag) == 0:
        supply_next_data_flag = None
    return next_current_list, blood_routine_dict, pathogenic_blood_dict, radio_dict, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict, supply_next_data_flag

def is_have_adverse_events(hadm_id, ill_time):
    if hadm_id in df_ae.index:
        filtered_rows = df_ae.loc[hadm_id, :]
        if not filtered_rows.empty:
            ae_time = filtered_rows[CHARTTIME]
            # ae_time = filtered_rows.iloc[hadm_id]["CHARTTIME"]
            ae_time = pd.to_datetime(ae_time)
            ill_time = pd.to_datetime(ill_time)
            if ill_time >= ae_time:
                return True
    return False


def select_period_csv():
    df_subject = pd.read_csv(
        TO_ROOT + 'preprocess/mimiciv_ill_redio_subject.csv',
        usecols=[SUBJECT_ID, HADM_ID, '性别', '年龄', MORTA_90d, ADMITTIME, ILL_TIME], encoding='gbk')
    print('正在多线程计算中--------------  ')
    pool = Pool(processes=20)

    pool.map(process_row, [(index, row, df_subject) for index, row in df_subject.iterrows()])
    pool.close()
    pool.join()


def process_row(args):
    index, row, df_subject = args
    print('==========================')
    print(f'患者索引：{index}')
    subject_id = row[SUBJECT_ID]
    hadm_id = row[HADM_ID]
    admittime = pd.to_datetime(row[ADMITTIME])
    ill_time = pd.to_datetime(row[ILL_TIME])

    #这里的时间范围不会影响补充数据有效期时间
    df_charts_illtime_before = get_endtime_before(df_charts, subject_id, hadm_id, ill_time + timedelta(hours=18))
    df_input_illtime_before = get_endtime_before(df_input, subject_id, hadm_id, ill_time + timedelta(hours=18))
    df_output_illtime_before = get_endtime_before(df_output, subject_id, hadm_id, ill_time + timedelta(hours=18))
    df_microbiology_illtime_before = get_endtime_before(df_microbiology, subject_id, hadm_id, ill_time + timedelta(hours=18))
    # df_radio_report_illtime_before = get_endtime_before(df_radio_report, subject_id, hadm_id,
    #                                                     ill_time + timedelta(hours=15))
    df_procedure_illtime_before = get_endtime_before(df_procedure, subject_id, hadm_id, ill_time + timedelta(hours=18))
    df_subject_diagnose = df_diagnose[df_diagnose[SUBJECT_ID] == subject_id]

    temperature_dict = get_temperature(df_charts_illtime_before)
    blood_pressure_dict = get_blood_pressure(df_charts_illtime_before)
    heart_rate_dict = get_heart_rate(df_charts_illtime_before)
    respiratory_rate_dict = get_respiratory_rate(df_charts_illtime_before)
    consciousness_dict = get_consciousness(df_charts_illtime_before)
    blood_pressure_systolic_dict = get_blood_pressure_systolic(df_charts_illtime_before)
    input_dict = get_input(df_input_illtime_before)
    output_dict = get_output(df_output_illtime_before)
    blood_routine_dict = get_blood_routine(df_charts_illtime_before)
    pathogenic_blood_dict = get_pathogenic_blood(df_charts_illtime_before, df_microbiology_illtime_before,df_subject_diagnose, ill_time + timedelta(hours=18))
    # radio_dict = get_radio(df_radio_report_illtime_before)
    radio_dict = None
    gas_analysis_dict = get_gas_analysis(df_charts_illtime_before)
    hemostasis_dict = get_hemostasis(df_charts_illtime_before)
    culture_dict = get_culture(df_charts_illtime_before, df_microbiology_illtime_before,df_procedure_illtime_before)
    smear_dict = get_smear(df_charts_illtime_before, df_subject_diagnose, ill_time + timedelta(hours=18))

    df = df_subject.loc[[index]].copy()
    df['体重'] = get_weight(df_charts_illtime_before)

    print(f'subject_id:{subject_id},hadm_id:{hadm_id},正在统计患病时间前后3h内的数据分布')
    if ill_time - admittime > pd.Timedelta(hours=3):
        starttime,endtime = get_illtime_from_admittime(admittime, ill_time)
        get_sample(df,'3h', ill_time,starttime-pd.Timedelta(hours=3), starttime,subject_id, hadm_id, temperature_dict, blood_pressure_dict,
                heart_rate_dict, input_dict, output_dict, respiratory_rate_dict, consciousness_dict,
                blood_pressure_systolic_dict, blood_routine_dict, pathogenic_blood_dict,
                radio_dict, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict)
        get_sample(df, '0h',ill_time, starttime,endtime, subject_id,
                   hadm_id, temperature_dict, blood_pressure_dict,
                   heart_rate_dict, input_dict, output_dict, respiratory_rate_dict, consciousness_dict,
                   blood_pressure_systolic_dict, blood_routine_dict, pathogenic_blood_dict,
                   radio_dict, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict)
        get_sample(df, '-3h',ill_time, endtime, endtime+pd.Timedelta(hours=3), subject_id, hadm_id, temperature_dict,
                   blood_pressure_dict, heart_rate_dict, input_dict, output_dict, respiratory_rate_dict, consciousness_dict,
                   blood_pressure_systolic_dict, blood_routine_dict, pathogenic_blood_dict,
                   radio_dict, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict)

def get_illtime_from_admittime(admittime,ill_time):
    ill_time = pd.to_datetime(ill_time)
    admittime = pd.to_datetime(admittime)
    for i in range(336):
        starttime = admittime + pd.Timedelta(hours=i+1)
        endtime = starttime+ pd.Timedelta(hours=1)
        if ill_time>= starttime and ill_time<= endtime:
            print(starttime)
            print(endtime)
            return starttime,endtime
    return None,None


result_filname = TO_ROOT + 'preprocess/1-mimiciv_ill_redio_subject_data_distribute.csv'

def get_sample(df,time_period,ill_time,starttime_current, endtime_current, subject_id, hadm_id, temperature_dict, blood_pressure_dict,
            heart_rate_dict, input_dict, output_dict, respiratory_rate_dict, consciousness_dict,
            blood_pressure_systolic_dict, blood_routine_dict, pathogenic_blood_dict,
            radio_dict, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict):
    print(f'subject_id:{subject_id},hadm_id:{hadm_id},正在统计{time_period}的数据')

    temp_time = starttime_current
    while temp_time < endtime_current:
        start_time = temp_time
        end_time = start_time + timedelta(hours=1)
        if time_period != '-3h':
            current_list, temperature_dict_new, blood_pressure_dict_new, heart_rate_dict_new, input_dict_new, output_dict_new, qsofa_new, supply_base_data_flag, ishave_qsofa, qsofa_sc = get_base_current(
                subject_id, hadm_id, start_time, end_time, temperature_dict, blood_pressure_dict,
                heart_rate_dict, input_dict, output_dict, respiratory_rate_dict, consciousness_dict,
                blood_pressure_systolic_dict)
            next_current_list, blood_routine_dict_new, pathogenic_blood_dict_new, radio_dict_new, gas_analysis_dict_new, hemostasis_dict_new, culture_dict_new, smear_dict_new, supply_next_data_flag = get_next_current(
                subject_id, hadm_id, start_time, end_time, blood_routine_dict, pathogenic_blood_dict,
                radio_dict, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict)
            if current_list is not None and next_current_list is not None:
                print(f'BASE_CURRENT_SUM  {len(current_list)},NEXT_CURRENT_SUM  {len(next_current_list)}')
                if len(current_list) >= 5 and len(next_current_list)+1 >= 4:
                    print(f'BASE_CURRENT_SUM  {current_list},NEXT_CURRENT_SUM  {next_current_list}')
                    df["TIME_RANGE"] = time_period
                    df["START_ENDTIME"] = str(start_time) + '~' + str(end_time)
                    df["BASE_CURRENT_SUM"] = len(current_list)
                    df["NEXT_CURRENT_SUM"] = len(next_current_list)+ 1
                    df["IS_HAVE_QSOFA"] = ishave_qsofa
                    df["QSOFA_SCORE"] = qsofa_sc
                    if not os.path.exists(result_filname):
                        df.to_csv(result_filname, mode='w', index=False, quoting=csv.QUOTE_ALL, encoding='gbk')
                    else:
                        df.to_csv(result_filname, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL, encoding='gbk')
        else:
            #-3h
            # 发病时间没有使用抗生素的
            df_subject_input = df_input[df_input[CATEGORY] == 'Antibiotics']
            df_subject_input = df_subject_input[(df_subject_input[SUBJECT_ID] == int(subject_id))]
            df_subject_input[CHARTTIME] = pd.to_datetime(df_subject_input[CHARTTIME])
            mask = (df_subject_input[CHARTTIME] >= ill_time) & (
                    df_subject_input[CHARTTIME] < ill_time + timedelta(hours=3))
            df_subject_input = df_subject_input[mask]
            if len(df_subject_input) == 0:
                current_list, temperature_dict_new, blood_pressure_dict_new, heart_rate_dict_new, input_dict_new, output_dict_new, qsofa_new, supply_base_data_flag, ishave_qsofa, qsofa_sc = get_base_current(
                    subject_id, hadm_id, start_time, end_time, temperature_dict, blood_pressure_dict,
                    heart_rate_dict, input_dict, output_dict, respiratory_rate_dict, consciousness_dict,
                    blood_pressure_systolic_dict)
                next_current_list, blood_routine_dict_new, pathogenic_blood_dict_new, radio_dict_new, gas_analysis_dict_new, hemostasis_dict_new, culture_dict_new, smear_dict_new, supply_next_data_flag = get_next_current(
                    subject_id, hadm_id, start_time, end_time, blood_routine_dict, pathogenic_blood_dict,
                    radio_dict, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict)
                if current_list is not None and next_current_list is not None:
                    print(f'BASE_CURRENT_SUM  {len(current_list)},NEXT_CURRENT_SUM  {len(next_current_list)}')
                    if len(current_list) >= 5 and len(next_current_list)+1 >= 4:
                        print(f'BASE_CURRENT_SUM  {current_list},NEXT_CURRENT_SUM  {next_current_list}')
                        df["TIME_RANGE"] = time_period
                        df["START_ENDTIME"] = str(start_time) + '~' + str(end_time)
                        df["BASE_CURRENT_SUM"] = len(current_list)
                        df["NEXT_CURRENT_SUM"] = len(next_current_list)+1
                        df["IS_HAVE_QSOFA"] = ishave_qsofa
                        df["QSOFA_SCORE"] = qsofa_sc
                        if not os.path.exists(result_filname):
                            df.to_csv(result_filname, mode='w', index=False, quoting=csv.QUOTE_ALL, encoding='gbk')
                        else:
                            df.to_csv(result_filname, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL,
                                      encoding='gbk')
        temp_time = end_time


if __name__ == '__main__':
    print('Filtering for data-rich septic patients.')
    select_period_csv()
    print("end")
