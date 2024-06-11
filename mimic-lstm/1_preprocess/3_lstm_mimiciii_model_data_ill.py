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

column_list = ['subject_id', 'hadm_id','admittime','illtime','starttime','endtime','period', 'chronic airway obstruction',
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

# 加载Tscore模型需要的检查项数据
df_tscore_charts = pd.read_csv(TO_ROOT + 'tscore_train_val_test_ill_subject_tscore_charts.csv')
df_tscore_charts.loc[:, 'CHARTTIME'] = pd.to_datetime(df_tscore_charts['CHARTTIME'])
df_tscore_charts['VALUE'] = df_tscore_charts['VALUE'].replace({'Yes': 1, 'No': 0})
df_tscore_charts['VALUE'] = df_tscore_charts['VALUE'].replace({'+': ''})
mask = ~df_tscore_charts['LABEL'].str.contains('gcs', case=False)
df_tscore_charts.loc[mask, 'VALUE'] = pd.to_numeric(df_tscore_charts.loc[mask, 'VALUE'], errors='coerce')
df_tscore_charts = df_tscore_charts.dropna(subset=['VALUE'])

result_csv = TO_ROOT + 'lstm_ill_model_input_data_by_diff_meanvalue.csv'

# 准备患病患者 和 正常患者
df_ill_subject = pd.read_csv(TO_ROOT + '09-tscore_train_val_test_ill_subject_40.csv')
df_ill_subject.loc[:, 'ILL_TIME'] = pd.to_datetime(df_ill_subject['ILL_TIME'])
df_admittime = pd.read_csv(ROOT+'ADMISSIONS.csv',usecols=['HADM_ID','ADMITTIME'])
df_ill_subject = pd.merge(df_ill_subject,df_admittime,on=['HADM_ID'],how='left')

# 诊断表
df_diagnose = pd.read_csv(ROOT + 'DIAGNOSES_ICD.csv', usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
df_d_icd = pd.read_csv(ROOT + 'D_ICD_DIAGNOSES.csv', usecols=['ICD9_CODE', 'SHORT_TITLE'])
df_diagnose = pd.merge(df_diagnose, df_d_icd, on=['ICD9_CODE'], how='inner')
df_diagnose['SHORT_TITLE'] = df_diagnose['SHORT_TITLE'].str.lower()

df_ill_mean = pd.read_csv(TO_ROOT+'lstm_model_data_fill_meanvalue_tabel_ill.csv')
df_ill_mean['period_end_to_illtime'] = df_ill_mean['period_end_to_illtime'].astype(int)

def tscore_model_data():
    # 使用多进程
    pool = Pool(processes=20)
    pool.map(process_ill_row, [(index, row, df_tscore_charts) for index, row in df_ill_subject.iterrows()])
    pool.close()
    pool.join()
    # for index, row in df_ill_subject.iterrows():
    #     process_ill_row(index, row, df_tscore_charts)


# 先筛选该时间段的数据
# 第一次补0 用该患者的有效期内的检查值
# 第二次补0 用除了该患者 其他所有患者在相似位置（每个患者的患病时间之前相同距离）的平均值
def process_ill_row(args):
    index, row, df_tscore_charts = args
    subject_id = int(row['SUBJECT_ID'])
    hadm_id = int(row['HADM_ID'])
    admit_time = pd.to_datetime(row['ADMITTIME'])
    ill_time = pd.to_datetime(row['ILL_TIME'])
    print(f'-----正在准备索引为{index} 患病患者 SUBJECT_ID {subject_id},HADM_ID {hadm_id}数据中-----')
    gender = row['GENDER']

    df_sub_chart = df_tscore_charts[
        (df_tscore_charts['SUBJECT_ID'] == int(subject_id)) & (df_tscore_charts['HADM_ID'] == int(hadm_id))]
    df_sub_diagnose = df_diagnose[
        (df_diagnose['SUBJECT_ID'] == int(subject_id)) & (df_diagnose['HADM_ID'] == int(hadm_id))]
    df_subject_diagnose_set = set(df_sub_diagnose['SHORT_TITLE'])

    ill_end_time = find_time_period(admit_time,ill_time)
    temptime = admit_time

    while temptime < ill_end_time + pd.Timedelta(hours=3):

        start_time = temptime
        end_time = start_time + pd.Timedelta(hours=1)

        hour_to_illtime = int(round(((ill_time - end_time).total_seconds() / 3600)))

        period = None
        if start_time >= ill_end_time:
            period = '-3h'
        elif start_time<=ill_time and end_time>=ill_time:
            period = '0h'
        elif end_time >= ill_end_time-pd.Timedelta(hours=3) and end_time <= ill_end_time-pd.Timedelta(hours=1):
            period = '3h'
        elif end_time >= ill_end_time-pd.Timedelta(hours=12) and end_time <= ill_end_time-pd.Timedelta(hours=4):
            period = '12h'

        flag = is_have_adverse_events(subject_id, hadm_id, end_time)
        df_chart_current_period = df_sub_chart[(df_sub_chart['CHARTTIME'] < end_time) & (
                df_sub_chart['CHARTTIME'] >= start_time)]
        df_12h = df_sub_chart[(df_sub_chart['CHARTTIME'] < (end_time + timedelta(hours=12))) & (
                df_sub_chart['CHARTTIME'] >= (start_time - timedelta(hours=12)))]
        df_2h = df_sub_chart[(df_sub_chart['CHARTTIME'] < (end_time + timedelta(hours=2))) & (
                df_sub_chart['CHARTTIME'] >= (start_time - timedelta(hours=2)))]
        get_subject_charts(subject_id, hadm_id, df_chart_current_period, df_subject_diagnose_set, flag, df_12h, df_2h,
                           gender, start_time, end_time, admit_time,ill_time, period,hour_to_illtime)

        temptime = end_time

#通过入院时间和患病时间，找到从入院时间三小时一个窗口划到刚包含患病时间窗口的结束时间
def find_time_period(admin_time,ill_time):
    start_time = admin_time
    end_time = start_time + pd.Timedelta(hours=1)

    # Loop until 'ill_time' is within the interval
    while ill_time > end_time:
        start_time += pd.Timedelta(hours=1)
        end_time += pd.Timedelta(hours=1)

    return end_time

df_ae = pd.read_csv(TO_ROOT + 'adverse_events.csv')
df_ae = df_ae[~df_ae['CHARTTIME'].isna()]
df_ae.set_index(['HADM_ID'], inplace=True)


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
                       gender, start_time, end_time, admit_time,ill_time, time_period,hour_to_illtime):
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
                                    df_12h, start_time, end_time, time_period,hour_to_illtime,'alanine transaminase')
    amylase = get_final_chart_value(['Amylase'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                    end_time, time_period,hour_to_illtime,'amylase')
    art_ph = get_final_chart_value(['Arterial pH'], df_chart_current_period,  ae_flag, df_2h, start_time,
                                   end_time, time_period,hour_to_illtime,'arterial pH')
    aspartate = get_final_chart_value(['Asparate Aminotransferase (AST)'], df_chart_current_period,  ae_flag,
                                      df_12h, start_time, end_time, time_period,hour_to_illtime,'aspartate aminotransferase')
    bicarbon = get_final_chart_value(['Bicarbonate'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                     end_time, time_period,hour_to_illtime,'bicarbonate')
    bun = get_final_chart_value(['BUN'], df_chart_current_period,  ae_flag, df_12h, start_time, end_time,
                                time_period,hour_to_illtime,'bun')
    dbp = get_final_chart_value(['Blood Pressure Diastolic'], df_chart_current_period,  ae_flag, df_2h,
                                start_time, end_time, time_period,hour_to_illtime,'dbp')
    fio2 = get_final_chart_value(['FiO2'], df_chart_current_period,  ae_flag, df_12h, start_time, end_time,
                                 time_period,hour_to_illtime,'fio2')
    hr = get_final_chart_value(['Heart Rate'], df_chart_current_period,  ae_flag, df_2h, start_time, end_time,
                               time_period,hour_to_illtime,'heart rate')
    hematocrit = get_final_chart_value(['Hematocrit'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                       end_time, time_period,hour_to_illtime,'hematocrit')
    hemoglobin = get_final_chart_value(['Hemoglobin'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                       end_time, time_period,hour_to_illtime,'hemoglobin')
    lactate = get_final_chart_value(['Lactate'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                    end_time, time_period,hour_to_illtime,'lactate')
    lipase = get_final_chart_value(['Lipase'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                   end_time, time_period,hour_to_illtime,'lipase')
    os1 = get_final_chart_value(['Oxygen saturation'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                end_time, time_period,hour_to_illtime,'oxygen saturation')
    platelet = get_final_chart_value(['Platelets'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                     end_time, time_period,hour_to_illtime,'platelet count')
    potassium = get_final_chart_value(['Potassium'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                      end_time, time_period,hour_to_illtime,'potassium')
    rass = get_final_chart_value(['CAM-ICU RASS LOC'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                 end_time, time_period,hour_to_illtime,'rass')
    rr = get_final_chart_value(['Respiratory Rate'], df_chart_current_period,  ae_flag, df_2h, start_time,
                               end_time, time_period,hour_to_illtime,'respiratory rate')
    creatinine = get_final_chart_value(['Creatinine'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                       end_time, time_period,hour_to_illtime,'')
    sodium = get_final_chart_value(['Sodium'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                   end_time, time_period,hour_to_illtime,'sodium')
    sbp = get_final_chart_value(['Blood Pressure systolic'], df_chart_current_period,  ae_flag, df_2h,
                                start_time, end_time, time_period,hour_to_illtime,'sbp')
    wbc = get_final_chart_value(['WBC'], df_chart_current_period,  ae_flag, df_12h, start_time, end_time,
                                time_period,hour_to_illtime,'wbc')
    pao2 = get_final_chart_value(['PaO2'], df_chart_current_period,  ae_flag, df_12h, start_time, end_time,
                                 time_period,hour_to_illtime,'')
    map = get_final_chart_value(['MAP'], df_chart_current_period,  ae_flag, df_2h, start_time, end_time,
                                time_period,hour_to_illtime,'map')
    paco2 = get_final_chart_value(['Arterial PaCO2'], df_chart_current_period,  ae_flag, df_12h, start_time,
                                  end_time, time_period,hour_to_illtime,'arterial PaCO2')
    temperature = get_final_chart_value(['Temperature '], df_chart_current_period,  ae_flag, df_2h,
                                        start_time, end_time, time_period,hour_to_illtime,'temperature')
    weight = get_final_chart_value(
        ['Daily Weight', 'Admission Weight', 'Previous Weight', 'Present Weight', 'Weight Kg'], df_chart_current_period,
         ae_flag, df_12h, start_time, end_time, time_period,hour_to_illtime,'weight')
    gcs = get_final_chart_value(['GCS'], df_chart_current_period, ae_flag, df_2h, start_time, end_time,
                                time_period,hour_to_illtime,'gcs')

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
    row_list = [subject_id, hadm_id,str(admit_time),str(ill_time),str(start_time), str(end_time), time_period,
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


def get_final_chart_value(chart_name_list, df_chart_current_period, ae_flag, df_xxh, start_time, end_time,
                          time_period,hour_to_illtime,chart_column_name):
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


# 找当前时间范围内的检查项 或者 根据不良事件和检查项周期进行补充
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
                return df_xxh_chart.loc[df_xxh_chart['VALUE'].first_valid_index(), 'VALUE']
    else:
        return df_chart.loc[df_chart['VALUE'].first_valid_index(), 'VALUE']


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
