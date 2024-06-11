import numpy as np
import pandas as pd
from multiprocessing import Pool


PATH = '/home/ddcui/'
ROOT = f'{PATH}mimic-original-data/mimiciv_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'


def preprocess(df):
    df['temperature'] = df['temperature'].str.replace('°C', '')
    df['blood_pressure'] = df['blood_pressure'].str.replace('mmHg', '')
    df['heart_rate'] = df['heart_rate'].str.replace('bpm', '')
    df['output_sum'] = df['output_sum'].str.replace('ml', '')

    df['temperature'] = df['temperature'].apply(pd.to_numeric, errors='coerce')
    df['heart_rate'] = df['heart_rate'].apply(pd.to_numeric, errors='coerce')
    df['output_sum'] = df['output_sum'].apply(pd.to_numeric, errors='coerce')

    df['blood_pressure_high'] = df['blood_pressure'].str.split('/').str[0]
    df['blood_pressure_low'] = df['blood_pressure'].str.split('/').str[1]
    df['blood_pressure_low'] = df['blood_pressure_low'].astype(float)
    df['blood_pressure_high'] = df['blood_pressure_high'].astype(float)

    df.loc[(df['blood_pressure_high'] >= 90) & (df['blood_pressure_high'] <= 140)& (df['blood_pressure_low'] >= 60) & (df['blood_pressure_low'] <= 90), 'blood_pressure'] = '正常'
    df.loc[~((df['blood_pressure_high'] >= 90) & (df['blood_pressure_high'] <= 140)& (df['blood_pressure_low'] >= 60) & (df['blood_pressure_low'] <= 90)), 'blood_pressure'] = '不正常'


    df['temperature_flag'] = 0
    df.loc[df['temperature'] > 37.3, 'temperature_flag'] = '发热'
    df.loc[(df['temperature'] >= 36.1) & (df['temperature'] <= 37.3), 'temperature_flag'] = '正常'
    df.loc[df['temperature'] < 36.1, 'temperature_flag'] = '过低'

    df['heart_rate_flag'] = 0
    df.loc[(df['heart_rate'] > 100), 'heart_rate_flag'] = '过高'
    df.loc[(df['heart_rate'] >= 60) & (df['heart_rate'] <= 100), 'heart_rate_flag'] = '正常'
    df.loc[(df['heart_rate'] < 60) , 'heart_rate_flag'] = '过低'

    del df['blood_pressure_high']
    del df['blood_pressure_low']
    del df['heart_rate']
    del df['temperature']
    df = df.rename(columns={'heart_rate_flag':'heart_rate','temperature_flag':'temperature'})

    df['age'] = df['age'].apply(pd.to_numeric, errors='coerce')
    df['weight'] = df['weight'].apply(pd.to_numeric, errors='coerce')


    df['lymphocytes'] = df['lymphocytes'].str.replace('K/uL', '')
    # df['lymphocytes'] = df['lymphocytes'].str.replace('%', '')
    df['hemoglobin'] = df['hemoglobin'].str.replace('g/dL', '')
    df['pO2'] = df['pO2'].str.replace('mm Hg', '')
    df['ph'] = df['ph'].str.replace('units', '')
    df['hco3'] = df['hco3'].str.replace('mEq/L', '')
    df['fib'] = df['fib'].str.replace('mg/dL', '')
    df['plat'] = df['plat'].str.replace('K/uL', '')
    df['bilirubin'] = df['bilirubin'].str.replace('mg/dL', '')
    df['map'] = df['map'].str.replace('mmHg', '')
    df['creatinine'] = df['creatinine'].str.replace('mg/dL', '')
    df['ptt'] = df['ptt'].str.replace('sec', '')
    df['dopamine'] = df['dopamine'].str.replace('mg', '')
    df['dobutamine'] = df['dobutamine'].str.replace('mg', '')
    df['epinephrine'] = df['epinephrine'].str.replace('mg', '')
    df['norepinephrine'] = df['norepinephrine'].str.replace('mg', '')
    # df['lactate'] = df['lactate'].str.replace('mmol/L', '')

    df['lymphocytes'] = df['lymphocytes'].apply(pd.to_numeric, errors='coerce')
    df['hemoglobin'] = df['hemoglobin'].apply(pd.to_numeric, errors='coerce')
    df['crp'] = df['crp'].apply(pd.to_numeric, errors='coerce')
    df['pO2'] = df['pO2'].apply(pd.to_numeric, errors='coerce')
    df['ph'] = df['ph'].apply(pd.to_numeric, errors='coerce')
    df['hco3'] = df['hco3'].apply(pd.to_numeric, errors='coerce')
    df['fib'] = df['fib'].apply(pd.to_numeric, errors='coerce')
    df['plat'] = df['plat'].apply(pd.to_numeric, errors='coerce')
    df['bilirubin'] = df['bilirubin'].apply(pd.to_numeric, errors='coerce')
    df['map'] = df['map'].apply(pd.to_numeric, errors='coerce')
    df['creatinine'] = df['creatinine'].apply(pd.to_numeric, errors='coerce')
    df['inr'] = df['inr'].apply(pd.to_numeric, errors='coerce')
    df['ptt'] = df['ptt'].apply(pd.to_numeric, errors='coerce')
    df['dopamine'] = df['dopamine'].apply(pd.to_numeric, errors='coerce')
    df['dobutamine'] = df['dobutamine'].apply(pd.to_numeric, errors='coerce')
    df['epinephrine'] = df['epinephrine'].apply(pd.to_numeric, errors='coerce')
    df['norepinephrine'] = df['norepinephrine'].apply(pd.to_numeric, errors='coerce')
    # df['lactate'] = df['lactate'].apply(pd.to_numeric, errors='coerce')

    df['pct'] = df['crp'].apply(get_pct_by_crp)
    if 'act' in df.columns:
        del df['act']
    return df

def get_pct_by_crp(crp):
    if crp is None:
        return None
    crp = float(crp)
    if  crp< 87.6:
        pct = map_range(crp,25.3,87.6,0.09,0.3)
    elif crp > 52.9 and crp< 103.4:
        pct = map_range(crp,52.9,103.4,0.2,0.7)
    elif crp > 58.5 and crp< 132.4:
        pct = map_range(crp,58.5,132.4,0.6,2.0)
    elif crp > 69.7 and crp< 171.2:
        pct = map_range(crp,69.7,171.2,1.7,6.6)
    elif crp > 79.4 and crp< 174.6:
        pct = map_range(crp,79.4,174.6,1.4,5.2)
    elif crp > 60.9 and crp< 148.9:
        pct = map_range(crp,60.9,148.9,1.7,7.4)
    elif crp > 62.9:
        pct = map_range(crp,62.9,167.5,2.9,33.2)
    else:
        return None
    return pct

#按照crp比例算pct
def map_range(value, crp_min, crp_max, pct_min, pct_max):
    value = max(min(value, crp_max), crp_min)
    from_range = crp_max - crp_min
    to_range = pct_max - pct_min
    scaled_value = (value - crp_min) / from_range
    mapped_value = pct_min + scaled_value * to_range
    return mapped_value


def process_fill(kind):
    if kind == 'ill':
        df = pd.read_csv(TO_ROOT+'preprocess/8-mimiciv_ill_sample_fill_doctor_data_table.csv', encoding='gbk')
    else:
        df = pd.read_csv(TO_ROOT + 'preprocess/8-mimiciv_not_sample_fill_doctor_data_table.csv', encoding='gbk')

    df = preprocess(df)
    df['starttime'] = df['start_endtime'].str.split('~').str[0]
    df['endtime'] = df['start_endtime'].str.split('~').str[1]

    #读取sample的数据  选完sample之后又跑了一下这个数据
    if kind == 'ill':
        df_sample = pd.read_csv(TO_ROOT+'preprocess/9-mimiciv_ill_sample_15_chart_1500.csv', encoding='gbk')
    else:
        df_sample = pd.read_csv(TO_ROOT+'preprocess/9-mimiciv_not_sample_15_chart_1500.csv', encoding='gbk')
    #sample的这四列在生成前端数据的时候求了  这里就不求了
    df_sample['dopamine'] = None
    df_sample['dobutamine'] = None
    df_sample['epinephrine'] = None
    df_sample['norepinephrine'] = None
    # df_sample['lactate'] = None

    df_sample = preprocess(df_sample)

    pool = Pool(processes=10)
    results = pool.map(fill_row, [(index, row, df) for index, row in df_sample.iterrows()])
    pool.close()
    pool.join()
    df_result = pd.concat(results)
    print(df_result.head(10))
    if kind == 'ill':
        df_result.to_csv(TO_ROOT+'preprocess/10-mimiciv_ill_sample_15_chart_filled_1500.csv',encoding='gbk',index=False)
    else:
        df_result.to_csv(TO_ROOT+'preprocess/10-mimiciv_not_sample_15_chart_filled_1500.csv',encoding='gbk',index=False)

columns = ['lymphocytes', 'hemoglobin', 'crp', 'pO2', 'o2', 'ph', 'hco3', 'inr', 'fib', 'ptt', 'plat', 'bilirubin', 'map', 'creatinine', 'pct', 'dopamine', 'dobutamine', 'epinephrine', 'norepinephrine']


def fill_row(args):
    index,row,df = args
    subject_id = row['subject_id']
    hadm_id = row['hadm_id']
    gender = row['gender']
    age = row['age']
    weight = row['weight']
    temperature = row['temperature']
    blood_pressure = row['blood_pressure']
    heart_rate = row['heart_rate']
    qsofa_rr = row['qsofa_rr']
    qsofa_sbp = row['qsofa_sbp']
    qsofa_gcs = row['qsofa_gcs']
    output_sum = row['output_sum']
    for column in columns:
        value = str(row[column])
        if value == 'nan':
            df_column0 = df.dropna(subset=[column])
            df_column  = df_column0[(df_column0['gender'] == gender) & (df_column0['temperature'] == temperature) & (df_column0['blood_pressure'] == blood_pressure) & (df_column0['heart_rate'] == heart_rate) & (df_column0['qsofa_rr'] == qsofa_rr) & (
                df_column0['qsofa_sbp'] == qsofa_sbp) & (df_column0['qsofa_gcs'] == qsofa_gcs)]
            if len(df_column) == 0:
                value = df_column[column].mean()
                if str(value) == 'nan':
                    value = df[column].mean()
            elif len(df_column) == 1:
                value = df_column.iloc[0][column]
            else:
                df_sim = df_column[(df_column['age'] <= age + 5) & (df_column['age'] >= age - 5)]
                df_sim = df_sim[(df_sim['weight'] <= weight + 5) & (df_sim['weight'] >= weight - 5)]
                df_sim = df_sim[(df_sim['output_sum'] <= output_sum + output_sum * 0.1) & (df_sim['output_sum'] >= output_sum - output_sum * 0.1)]
                if len(df_sim) == 0:
                    value = df_column[column].mean()
                elif len(df_sim) == 1:
                    value = df_column.iloc[0][column]
                else:
                    df_same_age = df_sim[df_sim['age'] == age]
                    if len(df_same_age) == 0:
                        #没有同龄的取均值
                        value = df_sim[column].mean()
                    elif len(df_same_age) == 1:
                        value = df_same_age.iloc[0][column]
                    else:
                        value = df_same_age[column].mean()
        row[column] = value
        if str(value) == 'nan':
            print('存在没有补充的缺失值')
    df_row = row.to_frame().T
    return df_row

def check_sample():
    df_sample = pd.read_csv(TO_ROOT + 'preprocess/10-mimiciv_3000_sample.csv',usecols=['subject_id', 'admittime','ill_time','endtime'], encoding='gbk')
    df_sample['endtime'] = pd.to_datetime(df_sample['endtime'])

    df_doctor_have_data_ill = pd.read_csv(TO_ROOT + 'preprocess/8-mimiciv_ill_sample_fill_doctor_data_table.csv', encoding='gbk')
    df_doctor_have_data_ill['endtime'] = df_doctor_have_data_ill['start_endtime'].str.split('~').str[1]
    df_doctor_have_data_ill['starttime'] = df_doctor_have_data_ill['start_endtime'].str.split('~').str[0]
    df_doctor_have_data_not = pd.read_csv(TO_ROOT + 'preprocess/8-mimiciv_not_sample_fill_doctor_data_table.csv', encoding='gbk')
    df_doctor_have_data_not['endtime'] = df_doctor_have_data_not['start_endtime'].str.split('~').str[1]
    df_doctor_have_data_not['starttime'] = df_doctor_have_data_not['start_endtime'].str.split('~').str[0]

    df_doctor_have_data_ill['endtime'] = pd.to_datetime(df_doctor_have_data_ill['endtime'])
    df_doctor_have_data_not['endtime'] = pd.to_datetime(df_doctor_have_data_not['endtime'])

    df_sepsis = df_sample[df_sample['endtime'].isin(set(df_doctor_have_data_ill['endtime']))]
    def_not_sepsis = df_sample[df_sample['endtime'].isin(set(df_doctor_have_data_not['endtime']))]

    #把患病的多余的去掉，因为补充医生数据的时候直接以这个为准了  补充完之后 sample就可以直接根据hadm_id  endtime 直接读取了
    df_doctor_have_data_ill = df_doctor_have_data_ill[df_doctor_have_data_ill['endtime'].isin(df_sample['endtime'])]
    df_doctor_have_data_ill.to_csv(TO_ROOT + 'preprocess/9-mimiciv_ill_sample_15_chart_1500.csv',encoding='gbk',index=False)
    df_doctor_have_data_not.to_csv(TO_ROOT + 'preprocess/9-mimiciv_not_sample_15_chart_1500.csv',encoding='gbk',index=False)

    print(len(df_sepsis))
    print(len(def_not_sepsis))

if __name__ == "__main__":
    # check_sample()
    process_fill('ill')
    process_fill('not')