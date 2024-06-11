import pandas as pd
from src.constant import *


columns = ['PATIENT_CASE_ID','GROUP_ID','SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'GENDER', 'AGE', 'MARITAL_STATUS', 'RACE', 'SEPSIS_ONSET_TIME', 'CURRENT_TIME', 'SEPSIS_LABEL']


if __name__ == '__main__':
    df_sample_all = pd.read_csv(fn_sample_all, encoding='gbk')
    df_sample = df_sample_all.drop_duplicates(subset=['START_ENDTIME'])

    df = pd.DataFrame(columns=columns,dtype='object')
    patient_case_id = 1
    for index,row in df_sample.iterrows():
        row_data = []
        row_data.append(patient_case_id)
        row_data.append(row['GROUP'])
        row_data.append(row['SUBJECT_ID'])
        row_data.append(row['HADM_ID'])
        row_data.append(row['ADMITTIME'])
        demo_info = row['人口学信息']
        demo_dict = eval(demo_info)
        row_data.append(demo_dict['性别'])
        row_data.append(demo_dict['年龄'])
        row_data.append(demo_dict['婚姻状态'])
        row_data.append(demo_dict['种族'])
        row_data.append(row['ILL_TIME'])
        row_data.append(row['START_ENDTIME'])
        if str(row['ILL_TIME']) == 'nan':
            row_data.append(0)
        else:
            row_data.append(1)
        df.loc[index] = row_data
        patient_case_id +=1

    df.to_csv(root+'\\datasets\\Patient_Case_Info.csv',encoding='gbk',index=False)
