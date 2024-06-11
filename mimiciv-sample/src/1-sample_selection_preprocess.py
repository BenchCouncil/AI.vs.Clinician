# -*- coding: utf-8 -*-

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
import re


warnings.filterwarnings('ignore', category=FutureWarning)

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

MA_STATUS_MAP = {
    '': '未知（默认）',
    'WIDOWED': '丧偶',
    'SINGLE': '单身',
    'DIVORCED': '离异',
    'UNKNOWN (DEFAULT)': '未知（默认）',
    'SEPARATED': '分居',
    'MARRIED': '已婚',
    'LIFE PARTNER': '伴侣'
}
ETHNICITY_MAP = {
    'ASIAN - OTHER': '亚洲 - 其他',
    'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE': '美国印第安人/阿拉斯加原住民联邦认可部落',
    'WHITE - RUSSIAN': '白人 - 俄罗斯人',
    'HISPANIC/LATINO':'西班牙裔/拉丁裔',
    'HISPANIC/LATINO - PUERTO RICAN': '西班牙裔/拉丁裔 - 波多黎各人',
    'UNKNOWN/NOT SPECIFIED': '未知/未指定',
    'CARIBBEAN ISLAND': '加勒比岛屿',
    'ASIAN - VIETNAMESE': '亚洲 - 越南人',
    'SOUTH AMERICAN': '南美洲',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': '夏威夷原住民或其他太平洋岛民',
    'HISPANIC/LATINO - SALVADORAN': '西班牙裔/拉丁裔 - 萨尔瓦多人',
    'ASIAN': '亚洲',
    'WHITE': '白人',
    'UNABLE TO OBTAIN': '无法获取',
    'OTHER': '其他',
    'PORTUGUESE': '葡萄牙人',
    'BLACK/CAPE VERDEAN': '黑人/佛得角人',
    'WHITE - BRAZILIAN': '白人 - 巴西人',
    'ASIAN - ASIAN INDIAN': '亚洲 - 印度人',
    'ASIAN - KOREAN': '亚洲 - 韩国人',
    'HISPANIC/LATINO - MEXICAN': '西班牙裔/拉丁裔 - 墨西哥人',
    'BLACK/AFRICAN': '黑人/非洲人',
    'HISPANIC/LATINO - CUBAN': '西班牙裔/拉丁裔 - 古巴人',
    'HISPANIC/LATINO - COLOMBIAN': '西班牙裔/拉丁裔 - 哥伦比亚人',
    'HISPANIC/LATINO - GUATEMALAN': '西班牙裔/拉丁裔 - 危地马拉人',
    'ASIAN - JAPANESE': '亚洲 - 日本人',
    'HISPANIC/LATINO - DOMINICAN': '西班牙裔/拉丁裔 - 多米尼加人',
    'ASIAN - CHINESE': '亚洲 - 中国人',
    'MIDDLE EASTERN': '中东人',
    'MULTI RACE ETHNICITY': '多种族',
    'ASIAN - CAMBODIAN': '亚洲 - 柬埔寨人',
    'WHITE - OTHER EUROPEAN': '白人 - 其他欧洲人',
    'BLACK/HAITIAN': '黑人/海地人',
    'PATIENT DECLINED TO ANSWER': '患者拒绝回答',
    'ASIAN - THAI': '亚洲 - 泰国人',
    'WHITE - EASTERN EUROPEAN': '白人 - 东欧人',
    'HISPANIC OR LATINO': '西班牙裔或拉丁裔',
    'BLACK/AFRICAN AMERICAN': '黑人/非洲裔美国人',
    'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)': '西班牙裔/拉丁裔 - 中美洲（其他）',
    'HISPANIC/LATINO - HONDURAN': '西班牙裔/拉丁裔 - 洪都拉斯人',
    'ASIAN - FILIPINO': '亚洲 - 菲律宾人',
    'AMERICAN INDIAN/ALASKA NATIVE': '美洲印第安人/阿拉斯加原住民'
}



# 1.准备不良事件表
def adverse_events_part1():
    diagnose_list = ['acute kidney', 'dialysis', 'Acute myocardial infarction', 'cerebral hemorrhage', 'Transfusion',
                     'dialysis', 'emboli', 'fall', 'infection', 'difficile', 'antiemetic', 'hypotension', 'pneumonia',
                     'pneumothorax']
    diagnose_no = 'without mention of infection'
    icd_procedures = ['dialysis', 'mechanical ventilation', 'Transfusion', 'dialysis', 'Intubation']
    prescrption = ['Vitamin K', 'Benadryl', 'flumazenil', 'naloxone']

    df_diagnose = pd.read_csv(ROOT + TB_DIAG, usecols=[SUBJECT_ID, HADM_ID, ICD9_CODE])
    df_d_icd = pd.read_csv(ROOT + TB_D_DIAG, usecols=[ICD9_CODE, LONG_TITLE])
    df_diagnose = pd.merge(df_diagnose, df_d_icd, on=[ICD9_CODE], how='inner')
    contains_diagnose_no = df_diagnose[LONG_TITLE].str.contains(diagnose_no, case=False) | df_diagnose[
        LONG_TITLE].str.contains(diagnose_no, case=False)
    df_diagnose = df_diagnose[df_diagnose[[LONG_TITLE]].apply(
        lambda x: x.str.contains('|'.join(diagnose_list), case=False)).any(axis=1)]
    df_diagnose = df_diagnose[~contains_diagnose_no]
    df_diagnose = df_diagnose.drop_duplicates(subset=[SUBJECT_ID, HADM_ID])
    del df_diagnose[ICD9_CODE]
    del df_diagnose[LONG_TITLE]

    df_proce = pd.read_csv(ROOT + TB_PROCE_ICD, usecols=[SUBJECT_ID, HADM_ID, ICD9_CODE])
    df_icd_pro = pd.read_csv(ROOT + TB_D_PROCED, usecols=[ICD9_CODE, LONG_TITLE])
    df_procedures = pd.merge(df_proce, df_icd_pro, on=[ICD9_CODE], how='inner')
    filtered_df2 = df_procedures[df_procedures[[LONG_TITLE]].apply(
        lambda x: x.str.contains('|'.join(icd_procedures), case=False)).any(axis=1)]
    filtered_df2 = filtered_df2.drop_duplicates(subset=[SUBJECT_ID, HADM_ID])
    del filtered_df2[ICD9_CODE]
    del filtered_df2[LONG_TITLE]

    df_proce = pd.read_csv(ROOT + TB_PRESCR, usecols=[SUBJECT_ID, HADM_ID, STARTTIME, DRUG])
    df_proce = df_proce[
        df_proce[[DRUG]].apply(lambda x: x.str.contains('|'.join(prescrption), case=False)).any(axis=1)]
    df_proce.rename(columns={STARTTIME: CHARTTIME}, inplace=True)
    df_proce = drop(df_proce)
    del df_proce[DRUG]

    df_reinicu = pd.read_csv(ROOT + TB_ICUSTAY, usecols=[SUBJECT_ID, HADM_ID, ICU_ID])
    grouped = df_reinicu.groupby([SUBJECT_ID, HADM_ID]).size().reset_index(name='count')
    filtered_df4 = grouped[grouped['count'] > 1]
    del filtered_df4['count']

    df_item67 = pd.read_csv(ROOT + TB_MICRO, usecols=[SUBJECT_ID, HADM_ID, CHARTTIME, ORG_NAME])
    df_item67 = df_item67[df_item67[[ORG_NAME]].apply(
        lambda x: x.str.contains('|'.join(['difficile', 'pneumonia', 'DIALYSIS FLUID']), case=False)).any(axis=1)]
    df_item67 = drop(df_item67)
    del df_item67[ORG_NAME]

    df_item8 = pd.read_csv(ROOT + TB_ADMISS, usecols=[SUBJECT_ID, ADMITTIME])
    grouped = df_item8.groupby(SUBJECT_ID)[ADMITTIME].nunique()
    df_item8 = df_item8[~df_item8[SUBJECT_ID].isin(grouped[grouped == 1].index)]
    df_item8[ADMITTIME] = pd.to_datetime(df_item8[ADMITTIME])
    df_sorted = df_item8.groupby(SUBJECT_ID).apply(lambda x: x.sort_values(ADMITTIME)).reset_index(drop=True)
    df_diff = df_sorted.groupby(SUBJECT_ID).apply(lambda x: x[ADMITTIME].iloc[-1] - x[ADMITTIME].iloc[0]).dt.days
    df_filtered = df_item8[~df_item8[SUBJECT_ID].isin(df_diff[df_diff > 30].index)].sort_values(SUBJECT_ID)
    del df_filtered[ADMITTIME]

    df_concat = pd.concat([df_diagnose, filtered_df2], ignore_index=True, sort=False)
    df_concat = pd.concat([df_concat, df_proce], ignore_index=True, sort=False)
    df_concat = pd.concat([df_concat, filtered_df4], ignore_index=True, sort=False)
    df_concat = pd.concat([df_concat, df_item67], ignore_index=True, sort=False)
    df_concat = pd.concat([df_concat, df_filtered], ignore_index=True, sort=False)
    df_concat = drop(df_concat)
    df_concat[HADM_ID] = pd.to_numeric(df_concat[HADM_ID], errors='coerce').astype(int)
    print(df_concat)
    return df_concat

def drop(df):
    df[CHARTTIME] = pd.to_datetime(df[CHARTTIME])
    df_min_date = df.groupby([SUBJECT_ID, HADM_ID])[CHARTTIME].min().reset_index()
    df = df.merge(df_min_date, on=[SUBJECT_ID, HADM_ID, CHARTTIME], how='inner')
    df = df.drop_duplicates(subset=[SUBJECT_ID, HADM_ID, CHARTTIME])
    return df

def adverse_events():
    df_concat = adverse_events_part1()
    df_item128 = pd.read_csv(ROOT + TB_D_ITEM, usecols=[ITEMID, LABEL, LINKSTO])
    df_item128 = df_item128[df_item128[[LABEL]].apply(
        lambda x: x.str.contains('|'.join(['Transfusion', 'dialysis', 'Intubation']), case=False)).any(axis=1)]
    df_item128_group = df_item128.groupby([LINKSTO])
    df_all = pd.DataFrame()
    cols1 = [SUBJECT_ID, HADM_ID, CHARTTIME, ITEMID]
    cols2 = [SUBJECT_ID, HADM_ID, STARTTIME, ITEMID]
    for linksto, group_index in df_item128_group.groups.items():
        group = df_item128_group.get_group(linksto)
        if linksto == 'chartevents':
            for i, df_chart_chunk in enumerate(
                    pd.read_csv(ROOT + TB_CHART, usecols=cols1, iterator=True, chunksize=10000000)):
                df_chunk = pd.merge(group, df_chart_chunk, on=[ITEMID], how='inner')
                df_all = pd.concat([df_all, df_chunk], ignore_index=True, sort=False)
        elif linksto == 'outputevents':
            df = pd.read_csv(ROOT + TB_OUTPUT, usecols=cols1)
            df_merge = pd.merge(group, df, on=[ITEMID], how='inner')
            df_all = pd.concat([df_all, df_merge], ignore_index=True, sort=False)
        elif linksto == 'inputevents':
            df = pd.read_csv(ROOT + TB_INPUT, usecols=cols2)
            df.rename(columns={STARTTIME: CHARTTIME}, inplace=True)
            df_merge = pd.merge(group, df, on=[ITEMID], how='inner')
            df_all = pd.concat([df_all, df_merge], ignore_index=True, sort=False)
        elif linksto == 'procedureevents':
            df = pd.read_csv(ROOT + TB_PROCE, usecols=cols2)
            df.rename(columns={STARTTIME: CHARTTIME}, inplace=True)
            df_merge = pd.merge(group, df, on=[ITEMID], how='inner')
            df_all = pd.concat([df_all, df_merge], ignore_index=True, sort=False)
        elif linksto == 'datetimeevents':
            df = pd.read_csv(ROOT + TB_DATETIME, usecols=cols1)
            df_merge = pd.merge(group, df, on=[ITEMID], how='inner')
            df_all = pd.concat([df_all, df_merge], ignore_index=True, sort=False)
    df_all = drop(df_all)
    del df_all[ITEMID]
    del df_all[LABEL]
    del df_all[LINKSTO]
    df_all[HADM_ID] = pd.to_numeric(df_all[HADM_ID], errors='coerce').astype(int)
    print(df_all)
    df_concat = pd.concat([df_concat, df_all], ignore_index=True, sort=False)
    df_concat = drop(df_concat)

    if not os.path.exists(TO_ROOT + 'preprocess/'):
        os.makedirs(TO_ROOT + 'preprocess/')
    df_concat.to_csv(TO_ROOT + 'preprocess/mimiciv_adverse_events.csv', index=False,mode='w')


# 2.分离正常患者和患病患者 带ah to year字段
def get_ill_and_not_subject():
    # 用于限定正常患者和mimic iii不相交
    df_patient = pd.read_csv(ROOT + TB_PATIENT, usecols=[SUBJECT_ID,GENDER,AGE, YEAR_GROUP])
    df_patient[YEAR_GROUP] = df_patient[YEAR_GROUP].apply(lambda x: x.split(' - ')[0]).astype(str)
    df_patient[YEAR_GROUP] = df_patient[YEAR_GROUP].astype(int)

    df_admission = pd.read_csv(ROOT + TB_ADMISS, usecols=[SUBJECT_ID, HADM_ID, ADMITTIME, MA_STATUS, ETHNICITY])
    # print(f'mimiciv中住院患者总数 {len(df_admission.drop_duplicates(subset=[SUBJECT_ID]))}')
    df_icu = pd.read_csv(ROOT + TB_ICUSTAY, usecols=[SUBJECT_ID, HADM_ID, ICU_ID])
    df_radio = pd.read_csv(TO_ROOT + 'preprocess/mimic-cxr-2.0.0-metadata.csv', usecols=[SUBJECT_ID, STUDYDATE, DICOMID])

    df_sepsis_chort = pd.read_csv(TO_ROOT + 'preprocess/sepsis_cohort.csv', usecols=[ICU_ID,MORTA_90d, ILL_TIME])
    df_sepsis_chort = pd.merge(df_sepsis_chort, df_icu, on=[ICU_ID], how='inner')
    df_sepsis_radio = pd.merge(df_radio,df_sepsis_chort,on=[SUBJECT_ID],how='inner')
    df_sepsis_radio = pd.merge(df_sepsis_radio,df_patient,on=[SUBJECT_ID],how='left') #这里不能用内连接，因为患病患者不能限制2013年之后
    df_sepsis_radio = pd.merge(df_sepsis_radio, df_admission, on=[SUBJECT_ID, HADM_ID], how='inner')
    df_sepsis_radio = df_sepsis_radio.drop_duplicates(subset=[HADM_ID])

    # df_patient = df_patient[df_patient[YEAR_GROUP] >= 2013]
    df_not = df_icu[~df_icu[SUBJECT_ID].isin(df_sepsis_chort[SUBJECT_ID])]
    df_not = pd.merge(df_not, df_patient, on=[SUBJECT_ID], how='left')  # 这里不能用内连接，因为患病患者不能限制2013年之后
    df_not = pd.merge(df_not, df_admission, on=[SUBJECT_ID, HADM_ID], how='inner')
    df_not_radio = pd.merge(df_radio,df_not,on=[SUBJECT_ID],how='inner')
    df_not_radio_no = df_not[~df_not[SUBJECT_ID].isin(df_not_radio[SUBJECT_ID])]
    df_not_select = pd.concat([df_not_radio,df_not_radio_no.drop_duplicates(subset=[SUBJECT_ID]).sample(10000)])
    df_not_select = df_not_select.drop_duplicates(subset=[HADM_ID]) #如果用subject_id去重的话 正常患者只有14000了

    print(f'正常患者 {len(df_not_select.drop_duplicates(subset=[SUBJECT_ID]))}')
    #因为患病患者如果限制在2013年之后 只有才3500个患者左右，等后续从训练集中剔除吧
    print(f'患病患者（有影像图片） {len(df_sepsis_radio.drop_duplicates(subset=[SUBJECT_ID]))}')

    print('开始翻译人口学信息--婚姻状态、种族--宗教在mimiciv中被去掉了')
    df_sepsis_radio = dem_info(df_sepsis_radio)
    df_not_radio = dem_info(df_not_radio)
    if not os.path.exists(TO_ROOT + 'preprocess/'):
        os.makedirs(TO_ROOT + 'preprocess/')
    df_sepsis_radio.to_csv(TO_ROOT + 'preprocess/mimiciv_ill_redio_subject.csv', index=False, mode='w',encoding='gbk')
    df_not_radio.to_csv(TO_ROOT + 'preprocess/mimiciv_not_redio_subject.csv', index=False, mode='w',encoding='gbk')

# 3.人口学信息的翻译
def dem_info(df):
    df[GENDER] = df[GENDER].map({'F': '女', 'M': '男'})
    df[MA_STATUS] = df[MA_STATUS].map(MA_STATUS_MAP)
    df[ETHNICITY] = df[ETHNICITY].map(ETHNICITY_MAP)
    df = df.drop_duplicates(subset=[HADM_ID])
    df = df.rename(columns={GENDER: '性别', AGE: '年龄', MA_STATUS: '婚姻状态', ETHNICITY: '种族'})
    return df

# 4.准备正常患者和患病患者的chart 和 lab检查项  筛选和医生相关的检查项
def get_chart_lab_event():
    df_ill = pd.read_csv(TO_ROOT + 'preprocess/mimiciv_ill_redio_subject.csv',usecols=[HADM_ID],encoding='gbk')
    df_not = pd.read_csv(TO_ROOT + 'preprocess/mimiciv_not_redio_subject.csv',usecols=[HADM_ID],encoding='gbk')

    filename_ill = TO_ROOT + 'preprocess/mimiciv_ill_redio_subject_doctor_chart_lab.csv'
    filename_not = TO_ROOT + 'preprocess/mimiciv_not_redio_subject_doctor_chart_lab.csv'

    df_item = pd.read_csv(ROOT + TB_D_ITEM, usecols=[ITEMID, LABEL])
    missing_values0 = df_item[LABEL].isna()
    df_item = df_item[~missing_values0]
    df_labitem = pd.read_csv(ROOT + TB_D_LAB, usecols=[ITEMID, LABEL])
    missing_values1 = df_labitem[LABEL].isna()
    df_labitem = df_labitem[~missing_values1]

    columns = [SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE,VALUENUM, VALUEUOM]
    chart_list = ['Weight','past medical history','Temperature','Blood Pressure','Respiratory Rate','Heart Rate','Gcs','WBC', 'WBC Count', 'RBC', ' Rbc', 'RBC, Ascites', 'RBC, CSF', 'RBC, Joint Fluid', 'RBC, Pleural',
    'Hemoglobin', 'Absolute Hemoglobin', 'Hematocrit', 'Hematocrit, Ascites', 'Hematocrit, CSF',
    'Hematocrit, Joint Fluid', 'Hematocrit, Pleural', 'MCV', 'MCH', 'MCHC', 'Platelet Count', 'Platelets',
    'Atypical Lymphocytes', 'Lymphocytes, Percent', 'Lymphocytes', 'Monocytes', 'Neutrophils', 'Eosinophils',
    'Basophils', 'Absolute Lymphocyte Count', 'Nucleated Red Cells', 'Nucleated RBC', 'Large Platelets',
    'C-Reactive Protein',"Albumin", "Alveolar-arterial Gradient", "Base Excess", "Calculated Bicarbonate", "Calculated Total CO2",
    "Carboxyhemoglobin", "Chloride", "Creatinine", "Estimated GFR (MDRD equation)", "Free Calcium", "Glucose",
    "Hematocrit", "Hematocrit, Calculated", "Hemoglobin", "% Ionized Calcium", "Lactate", "Lithium", "Methemoglobin",
    "O2 Flow", "Osmolality", "Oxygen", "Oxygen Saturation", "P50 of Hemoglobin", "pCO2", "PEEP", "pH", "pH, Urine",
    "pO2", "Potassium", "Required O2", "Sodium, Body Fluid", "Sodium, Urine",
    "Sodium, Whole Blood", "Total Calcium", "WB tCO2",  "HCO3",
    "Inspired O2 Fraction", "FiO2ApacheIIValue",'PT', 'INR', 'Activated Clotting Time', 'Fibrinogen', 'Fibrinogen', 'D-Dimer','Culture','Smear',
                  'MAP','Bilirubin','Arterial Blood Pressure mean','ptt']

    print('reading df_labevents')
    for i, df_lab_chunk in enumerate(
            pd.read_csv(ROOT + TB_LAB, usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_lab_chunk, df_labitem, on=[ITEMID], how='inner')
        df_chunk = df_chunk[df_chunk[LABEL].str.contains('|'.join(chart_list), case=False)]

        df_chunk_ill = pd.merge(df_chunk, df_ill, on=[HADM_ID], how='inner')
        df_chunk_not = pd.merge(df_chunk, df_not, on=[HADM_ID], how='inner')
        to_csv(df_chunk_ill,filename_ill)
        to_csv(df_chunk_not,filename_not)

    print('reading df_chartevents')
    for i, df_chart_chunk in enumerate(
            pd.read_csv(ROOT + TB_CHART, usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_chart_chunk, df_item, on=[ITEMID], how='inner')
        df_chunk = df_chunk[df_chunk[LABEL].str.contains('|'.join(chart_list), case=False)]

        df_chunk_ill = pd.merge(df_chunk, df_ill, on=[HADM_ID], how='inner')
        df_chunk_not = pd.merge(df_chunk, df_not, on=[HADM_ID], how='inner')
        to_csv(df_chunk_ill, filename_ill)
        to_csv(df_chunk_not, filename_not)

#5.等sample选完之后，提取3000sample的所有 chart lab检查项
def get_sample_chart_lab_event():
    df_sample = pd.read_csv(TO_ROOT + 'front_end/10-mimiciv_3000_sample.csv',usecols=[HADM_ID],encoding='gbk')

    filename = TO_ROOT + 'front_end/mimiciv_3000_sample_all_chart_lab.csv'

    df_item = pd.read_csv(ROOT + TB_D_ITEM, usecols=[ITEMID, LABEL])
    missing_values0 = df_item[LABEL].isna()
    df_item = df_item[~missing_values0]
    df_labitem = pd.read_csv(ROOT + TB_D_LAB, usecols=[ITEMID, LABEL])
    missing_values1 = df_labitem[LABEL].isna()
    df_labitem = df_labitem[~missing_values1]

    columns = [SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE,VALUENUM, VALUEUOM]

    print('reading df_labevents')
    for i, df_lab_chunk in enumerate(
            pd.read_csv(ROOT + TB_LAB, usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_lab_chunk, df_labitem, on=[ITEMID], how='inner')
        df_chunk = pd.merge(df_chunk, df_sample, on=[HADM_ID], how='inner')
        to_csv(df_chunk,filename)

    print('reading df_chartevents')
    for i, df_chart_chunk in enumerate(
            pd.read_csv(ROOT + TB_CHART, usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_chart_chunk, df_item, on=[ITEMID], how='inner')
        df_chunk = pd.merge(df_chunk, df_sample, on=[HADM_ID], how='inner')
        to_csv(df_chunk, filename)

def to_csv(df,filename):
    df = df.reindex(columns=[SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE,VALUENUM, VALUEUOM,LABEL])
    df[HADM_ID] = df[HADM_ID].astype(int)
    if not os.path.exists(filename):
        df.to_csv(filename, mode='w', index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

#添加input中的肾上腺素 去肾上腺素 多巴胺  多巴酚丁胺 这些检查都在input中
def get_chart_lab_event_dopam():
    df_ill = pd.read_csv(TO_ROOT + 'preprocess/mimiciv_ill_redio_subject.csv',usecols=[HADM_ID],encoding='gbk')
    df_not = pd.read_csv(TO_ROOT + 'preprocess/mimiciv_not_redio_subject.csv',usecols=[HADM_ID],encoding='gbk')

    filename_ill = TO_ROOT + 'preprocess/mimiciv_ill_redio_subject_doctor_chart_lab.csv'
    filename_not = TO_ROOT + 'preprocess/mimiciv_not_redio_subject_doctor_chart_lab.csv'

    df_item = pd.read_csv(ROOT + TB_D_ITEM, usecols=[ITEMID, LABEL])
    missing_values0 = df_item[LABEL].isna()
    df_item = df_item[~missing_values0]

    columns = ['subject_id', 'hadm_id', 'itemid', 'starttime', 'amount','amountuom']
    chart_list = ['epinephrine','Dobutamine','dopamine']

    print('reading df_input')
    for i, df_chunk in enumerate(
            pd.read_csv(ROOT + TB_INPUT, usecols=columns, iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = pd.merge(df_chunk, df_item, on=[ITEMID], how='inner')
        df_chunk = df_chunk[df_chunk[LABEL].str.contains('|'.join(chart_list), case=False)]

        df_chunk_ill = pd.merge(df_chunk, df_ill, on=[HADM_ID], how='inner')
        df_chunk_not = pd.merge(df_chunk, df_not, on=[HADM_ID], how='inner')
        to_csv_input(df_chunk_ill,filename_ill)
        to_csv_input(df_chunk_not,filename_not)

def to_csv_input(df,filename):
    df = df.rename(columns={'starttime':CHARTTIME,'amount':VALUE,'amountuom':VALUEUOM})
    df[VALUENUM] = None
    df = df.reindex(columns=[SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUE,VALUENUM, VALUEUOM,LABEL])
    df[HADM_ID] = df[HADM_ID].astype(int)
    if not os.path.exists(filename):
        df.to_csv(filename, mode='w', index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


if __name__ == '__main__':
    print('正在生成mimiciv的不良事件表')
    adverse_events()

    print('根据mimic-cxr中的患者id（subject_id）和发病时间信息初步确定具有cxr的患病患者和正常患者id')
    get_ill_and_not_subject()

    print('提取上述患病患者和正常患者的chartevent和labevent中相应检查项')
    get_chart_lab_event()
    print('提取上述患病患者和正常患者的input中sofa相应检查项')
    get_chart_lab_event_dopam()

    print('end')
