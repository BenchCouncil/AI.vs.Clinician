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
import copy

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
ENDTIME = 'endtime'
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

#直接读3000sample的所有检查项  防止漏掉
df_charts = pd.read_csv(TO_ROOT + 'front_end/mimiciv_3000_sample_all_chart_lab.csv',
    usecols=[SUBJECT_ID,HADM_ID,ITEMID, CHARTTIME, VALUE,VALUENUM, VALUEUOM,LABEL])

df_charts[HADM_ID] = df_charts[HADM_ID].astype(str).str.strip().replace('\.0', '', regex=True)
df_charts[SUBJECT_ID] = df_charts[SUBJECT_ID].astype(int)
df_charts[HADM_ID] = df_charts[HADM_ID].astype(int)
df_charts.set_index([HADM_ID], inplace=True)
df_charts = df_charts[(~df_charts[VALUE].isna()) | (~df_charts[VALUE].isnull()) | ((df_charts[VALUE] != ''))]
df_charts = df_charts.drop_duplicates()

df_item = pd.read_csv(ROOT +TB_D_ITEM, usecols=[ITEMID,LABEL, CATEGORY])

df_input = pd.read_csv(ROOT + TB_INPUT,usecols=[SUBJECT_ID,HADM_ID,STARTTIME, ITEMID,AMOUNT,AMOUNTUOM, TOTALAMOUNT,RATE,RATEUOM])
df_input[AMOUNT] = pd.to_numeric(df_input[AMOUNT], errors='coerce').astype(float)
df_input = df_input.dropna(subset=[AMOUNT], how='any')
df_input.rename(columns={STARTTIME: CHARTTIME}, inplace=True)
df_input = pd.merge(df_input, df_item, on=[ITEMID], how='inner')
df_input.set_index([HADM_ID], inplace=True)

df_output = pd.read_csv(ROOT + TB_OUTPUT,usecols=[SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE, VALUEUOM])
df_output = pd.merge(df_output, df_item, on=[ITEMID], how='inner')
df_output_sofa = df_output[df_output[ITEMID].isin([226627, 226631])]
df_output_sofa[CHARTTIME] = pd.to_datetime(df_output_sofa[CHARTTIME])

df_output.set_index([HADM_ID], inplace=True)
df_output[VALUE] = pd.to_numeric(df_output[VALUE], errors='coerce').astype(float)
df_output = df_output.dropna(subset=[VALUE], how='any')


df_microbiology = pd.read_csv(ROOT + TB_MICRO)
df_micro = df_microbiology[df_microbiology['comments'] != 'nan']
df_cusm_zh = pd.read_csv(TO_ROOT+'front_end/16-culture_chinese.csv',encoding='gbk')
df_micro = pd.merge(df_micro,df_cusm_zh,on=['comments'],how='left')
df_culture = df_micro[(df_micro['spec_type_desc'].str.lower().str.contains('culture'))| (df_micro['test_name'].str.lower().str.contains('culture'))]
df_smear = df_micro[(df_micro['spec_type_desc'].str.lower().str.contains('smear'))| (df_micro['test_name'].str.lower().str.contains('smear'))]
df_microbiology.set_index([HADM_ID], inplace=True)
df_culture[CHARTTIME] = pd.to_datetime(df_culture[CHARTTIME])
df_smear[CHARTTIME] = pd.to_datetime(df_smear[CHARTTIME])

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

df_prescription = pd.read_csv(ROOT+'prescriptions.csv',usecols=[SUBJECT_ID,HADM_ID,STARTTIME,DRUG,'stoptime','dose_val_rx','dose_unit_rx'])
df_prescription['starttime'] = pd.to_datetime(df_prescription['starttime'])
df_prescription['stoptime'] = pd.to_datetime(df_prescription['stoptime'])
df_sofa_pres = df_prescription[df_prescription[['drug']].apply( lambda x: x.str.contains('|'.join(['Dopamine','Dobutamine','Epinephrine']), case=False)).any(axis=1)]
df_sofa_pres['drug'] = df_sofa_pres['drug'].str.lower()
df_prescription = df_prescription.rename(columns={STARTTIME: CHARTTIME})

df_prescr_zh = pd.read_csv(TO_ROOT+'front_end/14-all_drug_zh_final.csv',usecols=['drug_en','drug_zh_en'],encoding='gbk')
df_prescr_zh = df_prescr_zh.rename(columns={'drug_en': 'drug'})
df_prescription = pd.merge(df_prescription,df_prescr_zh,on=['drug'],how='left')

df_in_out = pd.read_csv(TO_ROOT+'front_end/15-input_output_chinese.csv',usecols=['drug_en','drug_zh_en'],encoding='gbk')
df_in_out_zh = pd.concat([df_prescr_zh,df_in_out])
df_prescription.set_index([HADM_ID], inplace=True)


df_ae = pd.read_csv(TO_ROOT + 'preprocess/mimiciv_adverse_events.csv')
df_ae = df_ae[~df_ae[CHARTTIME].isna()]
df_ae.set_index([HADM_ID], inplace=True)

#读取需要补充的数据
df_ill_fill = pd.read_csv(TO_ROOT+'front_end/10-mimiciv_ill_sample_15_chart_filled_1500.csv',encoding='gbk')
df_not_fill = pd.read_csv(TO_ROOT+'front_end/10-mimiciv_not_sample_15_chart_filled_1500.csv',encoding='gbk')
df_ill_fill[ENDTIME] = pd.to_datetime(df_ill_fill[ENDTIME])
df_not_fill[ENDTIME] = pd.to_datetime(df_not_fill[ENDTIME])

#机械通气
df_mechvent = pd.read_csv(TO_ROOT+'front_end/mimiciv_all_mech_vent_subject.csv')
df_mechvent[CHARTTIME] = pd.to_datetime(df_mechvent[CHARTTIME])

de_cxr = pd.read_csv(TO_ROOT+'front_end/5-mimiciv_sample_cxr_zh_frontend.csv',encoding='gbk',usecols=['subject_id','study_id','hadm_id','endtime','dicom_id','studytime','ProcedureCodeSequence_CodeMeaning','ViewCodeSequence_CodeMeaning','radio_report_zh'])
de_cxr = de_cxr.rename(columns={'studytime': CHARTTIME})
de_cxr['endtime'] = pd.to_datetime(de_cxr['endtime'])

df_note = pd.read_csv(TO_ROOT+'front_end/6-mimiciv_sample_note_zh_sort.csv',usecols=['hadm_id', 'charttime','endtime', 'text_zh', '影像类型', '影像部位'], encoding='gbk')
df_note['charttime'] = pd.to_datetime(df_note['charttime'])
df_note['endtime'] = pd.to_datetime(df_note['endtime'])

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

def get_start_to_endtime(df, starttime, endtime):

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
        df['temp'] = df.apply(
            lambda row: {'影像类型': str(row['ProcedureCodeSequence_CodeMeaning']),'影像部位': str(row['ViewCodeSequence_CodeMeaning']), '时间': str(row[CHARTTIME]), '影像图片':f'jpg_files/p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{str(row["dicom_id"])+".jpg"}','值': str(row['radio_report_zh'])},
            axis=1)
        time_series_list = df['temp'].tolist()
        del df['temp']
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
                    '入量': str(round(float(row[AMOUNT]),2)) + str(row[AMOUNTUOM]),
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
                lambda row: {key: value for key, value in {'出量': str(round(float(row[VALUE]),2)) + str(row[VALUEUOM]),
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
                lambda row: {str(row[LABEL]): str(row[VALUE]) + str(row[VALUEUOM])} if str(row[VALUE]) != '' else {}, axis=1)
            # group['temp'] = group.apply(
            #     lambda row: {str(row[LABEL]): str(row[VALUE]) + str(row[VALUEUOM])} if str(row[VALUE]) != '' else None, axis=1)
            # group = group[group['temp'] != None]
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
            df = df[df[LABEL] != '']
            df['temp'] = df.apply(lambda row: {'送检样本': str(row[LABEL]), '时间': str(row[CHARTTIME]),
                                               '值': str(row[VALUE]) + str(row[VALUEUOM]).replace('None', '')}, axis=1)
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
                                                                   '药物浓度与参考浓度对比': str(row[DILUTION_COMPARISON]).strip(),
                                                                   '抗生素敏感性时的稀释值': row[DILUTION_VALUE],
                                                                   '抗生素的敏感性和试验结果': row[INTERPRETATION]}.items()
                                     if value != ''}, axis=1)
                    group = group[group[SPEC_TYPE_DESC] != '']
                    spec_type_desc_values = list(group[SPEC_TYPE_DESC])
                    if len(spec_type_desc_values) > 0:
                        entry = {'时间': str(name), '送检样本': spec_type_desc_values[0], '值': group['temp1'].tolist()}
                        time_series_list1.append(entry)
                time_series_list.extend(time_series_list1[::-1])
    elif chartname == '涂片':
        if not df.empty:
            df = df[df[VALUE] != '']
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

def get_demographic_info(df_subject_demographic_info, df_charts_illtime_before,start_time,weight_row,hadm_id):
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

    if weight_series_dict is not None and int(hadm_id) != 24995393 and int(hadm_id) != 26061152:
        df_subject_demographic_info.update(weight_series_dict)
    else:
        weight_dict = {'体重': [{'时间': str(start_time), '值': str(weight_row)+'kg'}]}
        df_subject_demographic_info.update(weight_dict)
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

history_translation = {
    'Diabetes - Oral Agent': '糖尿病 - 口服药物',
    'nan': '无',  # 'nan' 表示缺失值，翻译为 "无"
    'CVA': '脑卒中',
    'MI': '心肌梗死',
    'Pacemaker': '心脏起搏器',
    'GI Bleed': '胃肠道出血',
    'Hepatitis': '肝炎',
    'CHF': '充血性心力衰竭',
    'Asthma': '哮喘',
    'Seizures': '癫痫',
    'ETOH': '酒精（饮酒）史',
    'Diabetes - Insulin': '糖尿病 - 胰岛素',
    'Arrhythmias': '心律失常',
    'Liver Failure': '肝功能衰竭',
    'Pancreatitis': '胰腺炎',
    'CAD': '冠心病',
    'Smoker': '吸烟者',
    'PVD': '外周血管病变',
    'Anemia': '贫血',
    'COPD': '慢性阻塞性肺疾病',
    'Renal Failure': '肾功能衰竭',
    'Hypertension': '高血压',
    'Angina': '心绞痛',
    'HEMO or PD': '血液透析或腹膜透析'
}

def get_ill_history(df_charts_illtime_before):
    ill_history_label_list = ['past medical history']
    df_ill_history = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(ill_history_label_list), case=False)).any(axis=1)]
    ill_history = df_ill_history[VALUE].tolist()
    #去重加汉化
    translated_history_set = {history_translation.get(item, item) for item in ill_history}
    translated_history_set = '\n'.join(translated_history_set)
    return translated_history_set

def get_cxr(hadm_id,endtime):
    df_sample = de_cxr[(de_cxr[HADM_ID] == hadm_id) & (de_cxr['endtime'] == endtime)]
    cxr_time_series_dict = get_time_series(df_sample, '影像报告')
    return cxr_time_series_dict

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
    df_temperature = df_temperature.drop_duplicates(subset=[CHARTTIME])
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
    df_heart_rate = df_heart_rate.drop_duplicates(subset=[CHARTTIME])
    heart_rate_series_dict = get_time_series(df_heart_rate, '心率')
    return heart_rate_series_dict

def get_respiratory_rate(df_charts_illtime_before):
    respiratory_rate_label_list = ['Respiratory Rate']
    df_respiratory_rate = df_charts_illtime_before[
        df_charts_illtime_before[[LABEL]].apply(
            lambda x: x.str.contains('|'.join(respiratory_rate_label_list), case=False)).any(axis=1)]
    df_respiratory_rate = df_respiratory_rate.drop_duplicates(subset=[CHARTTIME])
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
        df_consciousness = df_consciousness[df_consciousness[[VALUE]].apply(lambda x: x.str.contains('意识', case=False)).any(axis=1)]
    else:
        gcs_label = ['GCS - Eye Opening', 'GCS - Verbal Response', 'GCS - Motor Response']
        df_consciousness = process_gcs_label(df_charts_illtime_before,gcs_label)
        if len(df_consciousness) == 0:
            gcs_label = ['GCSEyeApacheIIValue', 'GCSMotorApacheIIValue', 'GCSVerbalApacheIIValue']
            df_consciousness = process_gcs_label(df_charts_illtime_before, gcs_label)
            if len(df_consciousness) == 0:
                gcs_label = ['GCSMotor_ApacheIV', 'GCSEye_ApacheIV', 'GCSVerbal_ApacheIV']
                df_consciousness = process_gcs_label(df_charts_illtime_before, gcs_label)

    consciousness_series_dict = get_time_series(df_consciousness, '意识')
    return consciousness_series_dict

def process_gcs_label(df_charts_illtime_before,gcs_label):
    df_consciousness = df_charts_illtime_before[df_charts_illtime_before[[LABEL]].apply(
        lambda x: x.str.contains('|'.join(gcs_label), case=False)).any(axis=1)]
    if len(df_consciousness) > 0 and (len(df_consciousness) % 3 == 0):
        df_consciousness = get_gcs(df_consciousness)
        df_consciousness[VALUE] = pd.to_numeric(df_consciousness[VALUE], errors='coerce')
        df_consciousness[VALUE] = np.where(df_consciousness[VALUE] <= 14, '意识改变', '意识正常')
        df_consciousness = df_consciousness[
            df_consciousness[[VALUE]].apply(lambda x: x.str.contains('意识', case=False)).any(axis=1)]
    return df_consciousness

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

def get_input_output_zh(dict_data,kind,kind_name):
    if dict_data is None:
        return dict_data
    list_value = dict_data[kind]
    for value in list_value:
        value_list = value['值']
        for name in value_list:
            name_en = name[kind_name]
            df_zh_en = df_in_out_zh[df_in_out_zh['drug_en'].str.lower() == name_en.lower()]
            if len(df_zh_en) > 0:
                name_zh_en = df_zh_en.iloc[0]['drug_zh_en']
                name[kind_name] = name_zh_en
    dict_data[kind] = list_value
    return dict_data

def get_blood_routine(df_charts_illtime_before):
    blood_routine_label_list = ['WBC','RBC', 'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC','Nucleated Red Cells', 'Platelet Count', 'Platelets',
     'Lymphocytes','Absolute Lymphocyte Count', 'Monocytes', 'Neutrophils', 'Eosinophils', 'Basophils', 'Absolute Lymphocyte Count', 'C-Reactive Protein','CRP','bun','Creatinine']
    blood_routine_dict = {
        'WBC': 'WBC|白细胞计数',
        'WBC Count': 'WBC|白细胞计数',
        'RBC': 'RBC|红细胞计数',
        ' Rbc': 'RBC|红细胞计数',
        'RBC, Ascites': 'RBC-Ascites|红细胞-腹液',
        'RBC, CSF': 'RBC-CSF|红细胞-脑脊液',
        'RBC, Joint Fluid': 'RBC-Joint Fluid|红细胞-关节液',
        'RBC, Pleural': 'RBC-Pleural|红细胞-胸膜',
        'MCV': 'MCV|红细胞平均体积',
        'Hemoglobin':'HGB|血红蛋白浓度',
        'MCH': 'MCH|平均血红蛋白量',
        'MCHC': 'MCHC|平均血红蛋白浓度',
        'Hematocrit': 'HCT|红细胞压积',
        'Hematocrit, Ascites': 'HCT-Ascites|红细胞压积-腹液',
        'Hematocrit, CSF': 'HCT-CSF|红细胞压积-脑脊液',
        'Hematocrit, Joint Fluid': 'HCT-Joint Fluid|红细胞压积-关节液',
        'Hematocrit, Pleural': 'HCT-Pleural|红细胞压积-胸膜',
        'Nucleated RBC': 'NRBC%|有核红细胞比值',
        'Nucleated Red Cells': 'NRBC%|有核红细胞比值',
        'Platelet Count': 'PLT|血小板计数',
        'Large Platelets': 'P_LCR|大型血小板比率',
        'Atypical Lymphocytes': 'ATL|非典型淋巴细胞',
        'Lymphocytes, Percent':'LYMPH|淋巴细胞比值',
        'Lymphocytes': 'LYMPH|淋巴细胞比值',
        'Absolute Lymphocyte Count': 'LYMPH#|淋巴细胞绝对值',
        'Monocytes': 'MONO|单核细胞比值',
        'Neutrophils': 'NEUT|中性粒细胞比值',
        'Eosinophils': 'EO|嗜酸性粒细胞比值',
        'Basophils': 'BASO|嗜碱性粒细胞比值',
        'C-Reactive Protein': 'CRP|C反应蛋白',
        'High-Sensitivity CRP': 'sCRP|超敏C反应蛋白',
        'Albumin': 'Albumin|白蛋白',
        'BUN': 'BUN|尿素氮',
        'Bun': 'BUN|尿素氮',
        'Creatinine': 'Cr|肌酐',
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
    blood_gas_analysis_label_list = ["Alveolar-arterial Gradient", "Base Excess", "Calculated Bicarbonate", "Calculated Total CO2",
    "Carboxyhemoglobin", "Chloride",  "Estimated GFR (MDRD equation)", "Calcium", "Glucose",
    "Hematocrit",  "Hemoglobin", "Lactate", "Lithium",'Potassium', "Methemoglobin",
     "Osmolality", "Oxygen", "P50 of Hemoglobin",  "PEEP", "pH",
    "Potassium", 'O2', 'Sodium', 'Temperature',  "HCO3 (serum)"]
    gas_analysis_dict = {
        'Alveolar-arterial Gradient':'A-a gradient|肺泡-动脉梯度',
        'Base Excess':'BE|剩余碱',
        'Calculated Bicarbonate':'Bicarbonate|碳酸氢盐',
        'Calculated Total CO2':'TCO2|总二氧化碳',
        'Carboxyhemoglobin': 'HbCO|碳氧血红蛋白',
        'Chloride': 'CL|氯离子',
        'Estimated GFR (MDRD equation)': 'GRF|肾小球滤过率',
        'Free Calcium': 'Free Ca|游离钙',
        'Glucose': 'GLU|葡萄糖',
        '% Ionized Calcium': 'Ionized Calcium|游离钙百分比',
        'Total Calcium': 'Total Calcium|总钙',
        'Lactate': 'LACT2|乳酸',
        'Potassium': 'K+|钾',
        'Lithium': 'Lithium|锂',
        'Methemoglobin': 'MHb|高铁血红蛋白',
        'O2 Flow': 'O2 Flow|氧气流量',
        'Osmolality': 'm0sm|渗透压',
        'Oxygen': 'Oxygen|氧气',
        'Oxygen Saturation': 'SO2%|血氧饱和度',
        'P50 of Hemoglobin': 'P50 of HB|血红蛋白的P50值',
        'pCO2': 'pCO2|二氧化碳分压',
        'PEEP': 'PEEP|呼气末正压',
        'pH': 'pH|酸碱度',
        'pH, Urine': 'pH Urine|尿液pH值',
        'pO2': 'pO2|氧分压',
        'pO2, Body Fluid': 'pO2 Body Fluid|体液氧分压',
        'Required O2': 'O2|所需氧气',
        'Sodium, Body Fluid': 'Sodium Body Fluid|体液中的钠',
        'Sodium, Urine': 'Sodium Urine|尿液中的钠',
        'Sodium, Whole Blood': 'Sodium Whole Blood|全血中的钠',
        'Temperature': 'T|体温',
        'WB tCO2': 'WB tCO2|全血总二氧化碳',
        'HCO3 (serum)': 'HCO3|碳酸氢根(血清)',
        'Inspired O2 Fraction': 'FIO2|吸入氧浓度',
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
    hemostasis_label_list = ['PT', 'INR', 'Fibrinogen', 'D-Dimer']
    hemostasis_dict = {
         'PT': 'PT|凝血酶原时间',
        'PTT': 'PTT|部分凝血激酶时间',
        'INR(PT)': 'INR|国际标准化比值',
        'INR': 'INR|国际标准化比值',
        'Fibrinogen': 'FIB|纤维蛋白原',
        'D-Dimer': 'D-D2(||)|D-二聚体（II）',
        'D-Dimer (SOFT)': 'D-D2(||)|D-二聚体（II）'
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

def get_his(df,kind):
    df = df.drop_duplicates()
    df['temp'] = df.apply(lambda row: {'送检样本': str(row['test_name']), '时间': str(row['charttime']),'值': str(row['comments_zh'])}, axis=1)
    time_series_list = df['temp'].tolist()
    time_series_dict = {kind: time_series_list}
    return time_series_dict

def culture_or_smear(df,kind,endtime):
    history = None
    current = None
    if len(df) == 0:
        return None,None
    if df.iloc[0]['charttime'] > endtime - pd.Timedelta(hours=2) :
        df_cur = df.head(1)
        df_cur['charttime'] = endtime - pd.Timedelta(hours=1)
        current = get_his(df_cur,kind)
    if len(df) > 1:
        history = get_his(df[1:],kind)
    return current,history
smear_dict_new = {
    'SMEAR FOR BACTERIAL VAGINOSIS':"细菌性阴道炎涂片检查",
    'STATE AFB CULTURE AND SMEAR': "抗酸杆菌（AFB）培养和涂片检查"
}
culture_dict_new = {
    'Aerobic Bottle Gram Stain': '厌氧瓶革兰氏染色',
    'RESPIRATORY CULTURE': '呼吸道培养',
    'STATE AFB CULTURE AND SMEAR': '抗酸杆菌（AFB）培养和涂片检查',
    'FECAL CULTURE - R/O YERSINIA': '粪便培养 - 排除耶尔森氏菌',
    'FLUID CULTURE': '液体培养',
    'Stem Cell Culture in Bottles': '干细胞培养在瓶中',
    'Respiratory Viral Culture': '呼吸道病毒培养',
    'Rapid Respiratory Viral Antigen Test': '快速呼吸道病毒抗原测试',
    'FUNGAL CULTURE': '真菌培养',
    'Myco-F Bottle Gram Stain': 'Myco-F瓶革兰氏染色',
    'Blood Culture, Routine': '血液培养，常规',
    'VARICELLA-ZOSTER CULTURE': '水痘-带状疱疹病毒培养',
    'VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS': '病毒培养：排除单纯疱疹病毒',
    'R/O GROUP B BETA STREP': '排除B群β链球菌',
    'URINE CULTURE': '尿液培养',
    'CYTOMEGALOVIRUS EARLY ANTIGEN TEST (SHELL VIAL METHOD)': '巨细胞病毒早期抗原测试（壳体法）',
    'FECAL CULTURE': '粪便培养',
    'Enterovirus Culture': '肠道病毒培养',
    'VIRAL CULTURE: R/O CYTOMEGALOVIRUS': '病毒培养：排除巨细胞病毒',
    'THROAT - R/O BETA STREP': '喉咙 - 排除β链球菌',
    'Respiratory Virus Identification': '呼吸道病毒鉴定',
    'Tissue Culture-Bone Marrow': '组织培养-骨髓',
    'Fluid Culture in Bottles': '液体培养在瓶中',
    'BRUCELLA BLOOD CULTURE': '布鲁氏菌血液培养',
    'ANAEROBIC CULTURE': '厌氧培养',
    'BLOOD/FUNGAL CULTURE': '血液/真菌培养',
    'WOUND CULTURE': '伤口培养',
    'BLOOD/AFB CULTURE': '血液/抗酸杆菌培养',
    'FECAL CULTURE - R/O VIBRIO': '粪便培养 - 排除弧菌',
    'NOCARDIA CULTURE': '诺卡迪亚培养',
    'VIRAL CULTURE': '病毒培养',
    'CAMPYLOBACTER CULTURE': '弯曲杆菌培养',
    'Sonication culture, prosthetic joint': '声波培养，假体关节',
    'Anaerobic Bottle Gram Stain': '厌氧瓶革兰氏染色',
    'LEGIONELLA CULTURE': '军团菌培养',
    'FECAL CULTURE - R/O E.COLI 0157:H7': '粪便培养 - 排除大肠杆菌0157：H7'
}

#培养和涂片以这个函数为准 这里的培养只汉化了当前sample数据，如果修改sample之后需要再进行汉化
def get_culture_smear(next_current_list,hadm_id,endtime):
    df_cu = df_culture[(df_culture['hadm_id'] == hadm_id) & (df_culture['charttime'] <= endtime)]
    df_sm = df_smear[(df_smear['hadm_id'] == hadm_id) & (df_smear['charttime'] <= endtime)]

    df_cu = df_cu.sort_values('charttime', ascending=False)
    df_sm = df_sm.sort_values('charttime', ascending=False)

    df_cu['test_name'] = df_cu['test_name'].map(culture_dict_new)
    df_sm['test_name'] = df_sm['test_name'].map(smear_dict_new)

    df_sm.dropna(subset=['comments_zh'], inplace=True)
    df_cu.dropna(subset=['comments_zh'], inplace=True)

    cu_current, cu_history = culture_or_smear(df_cu, '培养', endtime)
    sm_current, sm_history = culture_or_smear(df_sm, '涂片', endtime)

    next_current_after = []
    for nc in next_current_list:
        if '培养' not in nc.keys() and '涂片' not in nc.keys():
            next_current_after.append(nc)
    if cu_current is not None:
        next_current_after.append(cu_current)
    if sm_current is not None:
        next_current_after.append(sm_current)
    return next_current_after,cu_history,sm_history

drug_dict = {
    'dopamine':'多巴胺',
    'dobutamine' :'多巴酚丁胺',
    'epinephrine':'肾上腺素',
    'norepinephrine':'去甲肾上腺素'
}

def get_sofa_pre(df,weight):
    result_dict = {}
    df['dose_val_rx'] = pd.to_numeric(df['dose_val_rx'], errors='coerce')
    df = df.dropna(subset=['dose_val_rx'])

    grouped = df.groupby('drug')
    for drug_name, group in grouped:
        if drug_name in drug_dict.keys():
            key = drug_dict[drug_name]
            count = sum(group['dose_val_rx'])
            unit = group.iloc[0]['dose_unit_rx']

            time_diff = group['stoptime'].max() - group['starttime'].min()

            time_min = time_diff.total_seconds() / 60
            #先判断给药是否到达一个小时
            if time_min > 60:
                #计算ug/(kg*min)
                if str(unit) != 'nan' and str(round(float(count),2))!= 'nan':
                    value = round((count * 1000) / (float(weight) * float(time_min)), 2)
                    result_dict[key] = str(value)+'ug/(kg*min)'
    return result_dict


def get_urine(hadm_id,endtime):
    df_out = df_output_sofa[df_output_sofa[HADM_ID] == hadm_id]
    df_out = df_out[(df_out[CHARTTIME] <= endtime) & (df_out[CHARTTIME] >= endtime - pd.Timedelta(hours=24))]
    if len(df_out) > 0:
        urine = sum(df_out[VALUE])
        urine_unit = df_out.iloc[0][VALUEUOM]
        urine_result = str(round(urine, 2)) + urine_unit
        if urine_result != '0.0ml':
            return urine_result
    return None


def get_mechvent(hadm_id,endtime):
    df_subject_mv = df_mechvent[df_mechvent[HADM_ID] == hadm_id]
    if len(df_subject_mv) > 0:
        charttime = df_subject_mv[CHARTTIME].min()
        if endtime >= charttime:
            return True
    return False


def get_gcs_mimiciv(df_chart):
    df_chart1 = df_chart[(df_chart[ITEMID] == 226755) | (df_chart[ITEMID] == 227013)]
    if len(df_chart1) > 0:
        return get_gcs_by_group(df_chart1)
    else:
        df_chart2 = df_chart[df_chart[ITEMID].isin([220739, 223900, 223901])]
        if len(df_chart2) > 0:
            df_chart2[VALUE] = df_chart2[VALUENUM]
            if len(df_chart2) > 0 and (len(df_chart2) % 3 == 0):
                return get_gcs_by_group(df_chart2)
        else:
            df_chart3 = df_chart[df_chart[ITEMID].isin([226756, 226757, 226758])]
            if len(df_chart3) > 0 :
                df_chart3[VALUE] = df_chart3[VALUENUM]
                if len(df_chart3) > 0 and (len(df_chart3) % 3 == 0):
                    return get_gcs_by_group(df_chart3)
            else:
                df_chart4 = df_chart[df_chart[ITEMID].isin([227012, 227011, 227014])]
                if len(df_chart4) > 0:
                    df_chart4[VALUE] = df_chart4[VALUENUM]
                    if len(df_chart4) > 0 and (len(df_chart4) % 3 == 0):
                        return get_gcs_by_group(df_chart4)
    return None

def get_gcs_by_group(df_gcs):
    if len(df_gcs) > 0:
        grouped_df = df_gcs.groupby([CHARTTIME])
        for group_key, group_index in grouped_df.groups.items():
            group = grouped_df.get_group(group_key)
            count = sum(int(value) for value in group[VALUE])
            group_index_list = group_index.tolist()
            df_gcs.loc[group_index_list, VALUE] = count
        df_gcs = df_gcs.drop_duplicates(subset=[SUBJECT_ID, CHARTTIME])
        gcs = min(df_gcs[VALUE])
        return gcs
    return None


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
                chart = min(chart, key=lambda x: pd.to_datetime(x['时间']))

                chart['时间'] = str(start_time)
                current = {str_kind: chart}
                current_list.append(current)
                return current_list, dict_after, supply_data_flag
            else:
                if flag:
                    chart = [temp for temp in chart_dict[str_kind] if
                             (start_time)-pd.Timedelta(hours=3) <= pd.to_datetime(temp['时间']) <= start_time+ pd.Timedelta(hours=3)]
                    if len(chart) > 0:
                        mid_time =  start_time + (end_time - start_time) / 2
                        chart = min(chart, key=lambda x: abs(pd.to_datetime(x['时间']) - mid_time))
                        chart['时间'] = str(start_time)
                        current = {str_kind: chart}
                        current_list.append(current)
                        supply_data_flag.append(str_kind)
                    return current_list, dict_after, supply_data_flag
                else:
                    chart = [temp for temp in chart_dict[str_kind] if
                             (start_time_not_ae) <= pd.to_datetime(temp['时间']) <= end_time_not_ae]
                    if len(chart) > 0:
                        mid_time = start_time + (end_time - start_time) / 2
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
    next_current_list, radio_dict_after, supply_next_data_flag = getcurrent(supply_next_data_flag, flag, start_time,
                                                                      start_time - timedelta(days=7), end_time,
                                                                      end_time + timedelta(days=7),
                                                                      radio_dict, '影像报告', next_current_list)
    #影像报告特殊处理 有效期较长 添加最新的到当前信息中
    next_current_list = check_cxr(next_current_list, radio_dict['影像报告'], start_time, end_time)

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
    return next_current_list, blood_routine_dict, pathogenic_blood_dict, radio_dict_after, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict, supply_next_data_flag

def check_cxr(next_current_list,chart,start_time,end_time):
    has_cxr = any('影像报告' in d for d in next_current_list)
    if not has_cxr:
        mid_time = start_time + (end_time - start_time) / 2
        chart = min(chart, key=lambda x: abs(pd.to_datetime(x['时间']) - mid_time))
        chart['时间'] = str(start_time)
        current = {'影像报告': chart}
        next_current_list.append(current)
    return next_current_list


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



def to_sample_result_csv():
    #sepsis_label 有患病时间的就为1
    df_subject = pd.read_csv(TO_ROOT + 'front_end/10-mimiciv_3000_sample_add_ethnicity.csv',
        usecols=[SUBJECT_ID,HADM_ID,ADMITTIME,ILL_TIME,STARTTIME,ENDTIME,'性别','年龄','体重','婚姻状态','种族','morta_90','time_range','group','sepsis_label'], encoding='gbk')
    result_path = TO_ROOT+"front_end/mimiciv_sample_3000_front_end_display_data.csv"
    if os.path.exists(result_path):
        os.remove(result_path)
        print("上次文件已删除,重新生成")

    print(f'多线程处理mimiciv sample中，sample行数 {len(df_subject)}')
    pool = Pool(processes=10)
    pool.map(process_sample_result, [(index, row, df_subject) for index, row in df_subject.iterrows()])
    pool.close()
    pool.join()


def process_sample_result(args):
    index, row, df_subject = args
    subject_id = row[SUBJECT_ID]
    hadm_id = row[HADM_ID]

    time_range = row['time_range']
    start_time = pd.to_datetime(row[STARTTIME])
    end_time = pd.to_datetime(row[ENDTIME])
    start_endtime = str(row[STARTTIME]) + '~' + str(row[ENDTIME])
    group = row['group']
    ill_time = row[ILL_TIME]
    weight_row = row['体重']
    if str(ill_time) != 'nan':
        ill_time = pd.to_datetime(ill_time)
    demographic_dict = {'性别': row['性别'], '年龄': row['年龄'], '婚姻状态': row['婚姻状态'], '种族': row['种族']}

    df_charts_illtime_before = get_endtime_before(df_charts, subject_id, hadm_id, end_time + timedelta(hours=15))
    df_input_illtime_before = get_endtime_before(df_input, subject_id, hadm_id, end_time + timedelta(hours=15))
    df_output_illtime_before = get_endtime_before(df_output, subject_id, hadm_id, end_time + timedelta(hours=15))
    df_microbiology_illtime_before = get_endtime_before(df_microbiology, subject_id, hadm_id,end_time + timedelta(hours=15))
    df_procedure_illtime_before = get_endtime_before(df_procedure, subject_id, hadm_id, end_time + timedelta(hours=15))

    df_prescription_illtime_before = get_endtime_before(df_prescription, subject_id, hadm_id, end_time)
    df_prescription_illtime_before = df_prescription_illtime_before.dropna(subset=['drug_zh_en'])
    df_subject_diagnose = df_diagnose[(df_diagnose[SUBJECT_ID] == subject_id)]
    ill_history = get_ill_history(df_charts_illtime_before)
    temperature_dict = get_temperature(df_charts_illtime_before)
    blood_pressure_dict = get_blood_pressure(df_charts_illtime_before)
    heart_rate_dict = get_heart_rate(df_charts_illtime_before)
    respiratory_rate_dict = get_respiratory_rate(df_charts_illtime_before)
    consciousness_dict = get_consciousness(df_charts_illtime_before)
    blood_pressure_systolic_dict = get_blood_pressure_systolic(df_charts_illtime_before)
    gcs_num = get_gcs_mimiciv(df_charts_illtime_before)

    input_dict = get_input(df_input_illtime_before)
    output_dict = get_output(df_output_illtime_before)
    blood_routine_dict = get_blood_routine(df_charts_illtime_before)
    pathogenic_blood_dict = get_pathogenic_blood(df_charts_illtime_before, df_microbiology_illtime_before,df_subject_diagnose, end_time + timedelta(hours=15))
    radio_dict = get_cxr(hadm_id,end_time)

    gas_analysis_dict = get_gas_analysis(df_charts_illtime_before)
    hemostasis_dict = get_hemostasis(df_charts_illtime_before)
    culture_dict = get_culture(df_charts_illtime_before, df_microbiology_illtime_before,
                               df_procedure_illtime_before)
    smear_dict = get_smear(df_charts_illtime_before, df_subject_diagnose, end_time + timedelta(hours=15))

    demographic_info = get_demographic_info(demographic_dict,get_endtime_before(df_charts, subject_id, hadm_id, end_time),start_time,weight_row,hadm_id)

    current_list, temperature_dict_new, blood_pressure_dict_new, heart_rate_dict_new, input_dict_new, output_dict_new, qsofa_new, supply_base_data_flag, ishave_qsofa, qsofa_sc = get_base_current(
        subject_id, hadm_id, start_time, end_time, temperature_dict, blood_pressure_dict,
        heart_rate_dict, input_dict, output_dict, respiratory_rate_dict, consciousness_dict,
        blood_pressure_systolic_dict)
    next_current_list, blood_routine_dict_new, pathogenic_blood_dict_new, radio_dict_new, gas_analysis_dict_new, hemostasis_dict_new, culture_dict_new, smear_dict_new, supply_next_data_flag = get_next_current(
        subject_id, hadm_id, start_time, end_time, blood_routine_dict, pathogenic_blood_dict,
        radio_dict, gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict)

    #历史药物
    history_medicine = list(set(df_prescription_illtime_before['drug_zh_en']))

    #添加sofa
    df_sofa_pres_hadmid = df_sofa_pres[(df_sofa_pres['hadm_id'] == hadm_id) & (df_sofa_pres['starttime'] <= end_time)]
    sofa_pres = get_sofa_pre(df_sofa_pres_hadmid,weight_row)
    urine = get_urine(hadm_id,end_time)
    is_mechevent = get_mechvent(hadm_id, end_time)
    gcs_num = check_gcs(current_list, gcs_num)
    respiration, coagulation, liver, cardiovascular, cns, renal,pct = getsofa_chart(hadm_id, end_time,ill_time,sofa_pres,gcs_num,urine,is_mechevent)

    next_current_list = check_next_current(next_current_list, hadm_id, end_time, ill_time)
    if str(ill_time) == 'nan':
        time_range = None
    if len(history_medicine) == 0:
        history_medicine = None

    radio_dict_new = add_note(radio_dict_new, hadm_id, start_time,end_time)
    next_current_list_after,cu_history,sm_history = get_culture_smear(next_current_list,hadm_id, end_time)
    #输入输出汉化
    input_after = get_input_output_zh(input_dict_new, '输入', '补液名称')
    output_after = get_input_output_zh(output_dict_new, '输出', '出液名称')
    to_csv(subject_id, hadm_id,index, ill_time, time_range, start_time,start_endtime, group,demographic_info, ill_history, current_list, supply_base_data_flag,
           temperature_dict_new, blood_pressure_dict_new, heart_rate_dict_new, input_after, output_after,
           qsofa_new,next_current_list_after, supply_next_data_flag, blood_routine_dict_new, pathogenic_blood_dict_new,
           radio_dict_new,gas_analysis_dict_new, hemostasis_dict_new, cu_history, sm_history,pct,history_medicine,respiration, coagulation, liver, cardiovascular, cns, renal)

def getsofa_chart(hadm_id,endtime,ill_time,sofa_pres,gcs_num,urine,is_mechevent):
    if str(ill_time) == 'nan':
        df = df_not_fill
    else:
        df = df_ill_fill
    df = df[(df[HADM_ID] == hadm_id) & (df[ENDTIME] == endtime)]

    pO2 = df.iloc[0]['pO2']
    o2 = df.iloc[0]['o2']
    pO2_o2 = int((pO2/o2)*100)
    respiration = {'Pao2/FiO2': str(pO2_o2)+'mmHg', '机械通气': is_mechevent, '所属类别': '动脉血气分析'}

    plat = int(df.iloc[0]['plat'])
    coagulation = {'血小板': str(plat)+'K/uL', '所属类别': '血常规'}

    bilirubin = round(float(df.iloc[0]['bilirubin']),2)
    liver = {'胆红素':str(bilirubin)+'mg/dl'}

    map = round(float(df.iloc[0]['map']),2)
    cardiovascular = {'MAP': str(map)+'mmHg'}
    if sofa_pres is not None:
        cardiovascular.update(sofa_pres)

    cns={'gcs':gcs_num}

    creatinine = df.iloc[0]['creatinine']
    renal = {'肌 酐': str(round(float(creatinine),1))+'mg/dl'}
    if urine is not None:
        renal['尿量'] = urine

    pct = df.iloc[0]['pct']
    pct_value = str(round(float(pct),2)) + 'ng/ml'
    return respiration, coagulation, liver, cardiovascular, cns, renal,pct_value

def check_next_current(next_current_list,hadm_id,endtime,ill_time):
    if str(ill_time) == 'nan':
        df = df_not_fill
    else:
        df = df_ill_fill
    df = df[(df[HADM_ID] == hadm_id) & (df[ENDTIME] == endtime)]

    lymphocytes = df.iloc[0]['lymphocytes']
    route_1 = {'LYMPH#|淋巴细胞绝对值': str(round(float(lymphocytes),2)) + 'K/uL'}
    hemoglobin = df.iloc[0]['hemoglobin']
    route_2 = {'HGB|血红蛋白浓度': str(round(float(hemoglobin),2)) + 'g/dL'}
    crp = df.iloc[0]['crp']
    route_3 = {'CRP|C反应蛋白': str(round(float(crp), 2)) + 'mg/L'}

    hco3 = df.iloc[0]['hco3']
    gas_1 = {'HCO3|碳酸氢根(血清)': str(round(float(hco3), 2)) + 'mEq/L'}
    o2 = df.iloc[0]['o2']
    gas_2 = {'FIO2|吸入氧浓度': str(round(float(o2), 2)) + 'mg/L'}
    pO2 = df.iloc[0]['pO2']
    gas_3 = {'pO2|氧分压': str(round(float(pO2), 2)) + 'mmHg'}
    ph = df.iloc[0]['ph']
    gas_4 = {'pH|酸碱度': str(round(float(ph), 2)) + 'units'}
    lactate = df.iloc[0]['lactate']
    gas_5 = {'LACT2|乳酸': str(round(float(lactate), 2)) + 'mmol/L'}


    ptt = df.iloc[0]['ptt']
    blood_1 = {'PTT|部分凝血激酶时间': str(round(float(ptt), 2)) + 'sec'}
    inr = df.iloc[0]['inr']
    blood_2 = {'INR|国际标准化比值': str(round(float(inr), 2))}
    fib = df.iloc[0]['fib']
    blood_3 = {'FIB|纤维蛋白原': str(round(float(fib), 2)) + 'mg/dL'}

    route_fill_dict = {'LYMPH#|淋巴细胞绝对值': route_1, 'HGB|血红蛋白浓度': route_2, 'CRP|C反应蛋白': route_3}
    next_current_list = fill_kind(next_current_list, '血常规', route_fill_dict)

    gas_fill_dict = {'HCO3|碳酸氢根(血清)': gas_1, 'FIO2|吸入氧浓度': gas_2, 'pO2|氧分压': gas_3,'pH|酸碱度':gas_4,'LACT2|乳酸':gas_5}
    next_current_list = fill_kind(next_current_list, '动脉血气分析', gas_fill_dict)

    blood_fill_dict = {'PTT|部分凝血激酶时间': blood_1, 'INR|国际标准化比值': blood_2, 'FIB|纤维蛋白原': blood_3}
    next_current_list = fill_kind(next_current_list, '止凝血', blood_fill_dict)

    return next_current_list

def fill_kind(current_list,kind,kind_fill_dict):
    dict_first = current_list[0]
    dict_first_key = list(dict_first.keys())[0]
    current_time = dict_first[dict_first_key]['时间']

    has_route = any(kind in d for d in current_list)
    if has_route:
        for index, chart_dict in enumerate(current_list):
            if kind in chart_dict:
                route_dict = chart_dict[kind]
                route_value_list = route_dict['值']
                for route_key in list(kind_fill_dict.keys()):
                    has_key = any(route_key in d for d in route_value_list)
                    if not has_key:
                        route_value_list.append(kind_fill_dict[route_key])
                route_dict['值'] = route_value_list
                chart_dict[kind] = route_dict
                current_list[index] = chart_dict
    else:
        value_list = list(kind_fill_dict.values())
        new_dict = {kind: {'时间': current_time, '值': value_list}}
        current_list.append(new_dict)
    return current_list

def check_gcs(current_list,gcs_num):
    for current_dict in current_list:
        if 'QSOFA' in current_dict:
            qsofa_dict = current_dict['QSOFA']
            gcs_value = qsofa_dict['意识']
            if gcs_num is None:
                if gcs_value == '意识改变':
                    return 14
                else:
                    return 15
            else:
                if gcs_value == '意识改变' and int(gcs_num) > 14 :
                    return 14
                if gcs_value == '意识正常' and int(gcs_num) <= 14 :
                    return 15
    return gcs_num


def add_note(radio_dict,hadm_id,starttime,endtime):
    df_note_hadmid = df_note[(df_note['hadm_id'] == hadm_id) & (df_note['endtime'] == endtime)]
    df_note_hadmid = df_note_hadmid[df_note_hadmid['charttime'] < starttime]
    df_note_hadmid = df_note_hadmid.sort_values('charttime', ascending=False)
    df_note_hadmid['temp'] = df_note_hadmid.apply(
        lambda row: {'影像类型': str(row['影像类型']),
                     '影像部位': str(row['影像部位']), '时间': str(row['charttime']),
                     '值': str(row['text_zh'])}, axis=1)
    time_series_list = df_note_hadmid['temp'].tolist()
    if radio_dict is None:
        return {'影像报告':time_series_list}
    else:
        radio_dict_list = list(radio_dict['影像报告'])
        radio_dict_list.extend(time_series_list)
        return {'影像报告':radio_dict_list}

def to_csv(subject_id, hadm_id,index, ill_time, time_range,start_time, start_endtime,  group,
           demographic_info, ill_history, current_list, supply_base_data_flag, temperature_dict,
           blood_pressure_dict, heart_rate_dict, input_dict, output_dict, qsofa_new,
           next_current_list, supply_next_data_flag, blood_routine_dict, pathogenic_blood_dict, radio_dict,
           gas_analysis_dict, hemostasis_dict, culture_dict, smear_dict,pct,history_medicine,sofa_respiratory_system,
               sofa_coagulation_system,sofa_liver,sofa_cvd,sofa_gcs,sofa_kidney):
    my_list = ['0h', '3h']

    df = pd.DataFrame({
        'UNIQUE_ID':[index],
        'SUBJECT_ID': [subject_id],
        'HADM_ID': [hadm_id],
        'ILL_TIME': [ill_time],
        'TIME_RANGE': [time_range],
        'START_ENDTIME': [start_endtime],

        'GROUP': [group],
        '基础信息（当前）中补充的数据': [supply_base_data_flag],
        '下一步检查（当前）中补充的数据': [supply_next_data_flag],
        '医生ID': ['1--x'],
        '人口学信息': [demographic_info],
        '病史': [ill_history],
        'AI模型预测结果': [{'RANDOM': round(random.random(), 2), 'LSTM1_IsVisible': 'Yes',
                      'LSTM1_Predict_Time': random.choice(my_list)}],
        '基础信息（当前）': [current_list],
        '基础信息_体温（历史）': [check_history(temperature_dict, '体温',start_time)],
        '基础信息_血压（历史）': [check_history(blood_pressure_dict, '血压',start_time)],
        '基础信息_心率（历史）': [check_history(heart_rate_dict, '心率',start_time)],
        '基础信息_输入（历史）': [check_history(input_dict, '输入',start_time)],
        '基础信息_输出（历史）': [check_history(output_dict, '输出',start_time)],
        '基础信息_QSOFA（历史）': [qsofa_new],
        '下一步检查（当前）': [next_current_list],
        '下一步检查_血常规（历史）': [check_history(blood_routine_dict, '血常规',start_time)],
        '下一步检查_病原血检查（历史）': [check_history(pathogenic_blood_dict,'病原血检查',start_time)],
        '下一步检查_影像报告（历史）': [check_history(radio_dict, '影像报告',start_time)],
        '下一步检查_动脉血气分析（历史）': [check_history(gas_analysis_dict, '动脉血气分析',start_time)],
        '下一步检查_止凝血（历史）': [check_history(hemostasis_dict,'止凝血',start_time)],
        '下一步检查_培养（历史）': [check_history(culture_dict,'培养',start_time)],
        '下一步检查_涂片（历史）': [smear_dict],
        '降钙素原': [pct],
        '历史用药': [history_medicine],
        'SOFA_呼吸系统': [sofa_respiratory_system],
        'SOFA_凝血系统': [sofa_coagulation_system],
        'SOFA_肝脏': [sofa_liver],
        'SOFA_心血管系统': [sofa_cvd],
        'SOFA_中枢神经系统': [sofa_gcs],
        'SOFA_肾脏': [sofa_kidney] })

    filename = TO_ROOT + 'front_end/mimiciv_sample_3000_front_end_display_data.csv'
    if not os.path.exists(filename):
        df.to_csv(filename, mode='w', index=False, quoting=csv.QUOTE_ALL, encoding='gbk')
    else:
        df.to_csv(filename, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL, encoding='gbk')


#保证历史数据的时间要小于当前时间
def check_history(chart_dict1,chartname,starttime):
    if chart_dict1 is not None:
        chart_list = chart_dict1[chartname]
        chart_list_remove = copy.deepcopy(chart_list)
        if len(chart_list) == 0 :
            return None
        for index, chart_dict in enumerate(chart_list):
            value_list = [d for d in chart_dict['值'] if d]
            if len(value_list) == 0:
                chart_list_remove.remove(chart_dict)
            else:
                charttime = pd.to_datetime(chart_dict['时间'])
                if charttime >= pd.to_datetime(starttime):
                    chart_list_remove.remove(chart_dict)
        if len(chart_list_remove) == 0:
            return None
        else:
            if len(chart_list_remove) > 36:
                chart_list_remove = chart_list_remove[:36]
            chart_dict1[chartname] = chart_list_remove
    return chart_dict1

if __name__ == '__main__':

    to_sample_result_csv()
    print("end")
