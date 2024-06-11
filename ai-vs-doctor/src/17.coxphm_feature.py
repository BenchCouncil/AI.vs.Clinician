import pandas as pd
from src.constant import *

if __name__ == '__main__':
    df_tscore = pd.read_csv(root + '\\data\\model_feature\\tscore_mimiciv_sample_3000_group_12_input_data.csv')
    df_tscore.columns = df_tscore.columns.str.upper()

    df_patient_case = pd.read_csv(root + '\\datasets\\Patient_Case_Info.csv', encoding='gbk')
    df_patient_case['START_TIME'] = df_patient_case['CURRENT_TIME'].str.split('~').str[0]

    df_patient_case['START_TIME'] = pd.to_datetime(df_patient_case['START_TIME'])

    for index, row in df_tscore.iterrows():
        row_data = []
        df_patient = df_patient_case[df_patient_case['START_TIME'] == pd.to_datetime(row['START_TIME'])]
        patient_case_id = df_patient.iloc[0]['PATIENT_CASE_ID']

        df_tscore.at[index, 'PATIENT_CASE_ID'] = patient_case_id

    del df_tscore['SUBJECT_ID']
    del df_tscore['HADM_ID']
    del df_tscore['START_TIME']
    del df_tscore['END_TIME']

    first_column = df_tscore.pop('PATIENT_CASE_ID')
    df_tscore.insert(0, 'PATIENT_CASE_ID', first_column)
    df_tscore.to_csv(root + '\\datasets\\CoxPHM_Feature.csv', encoding='gbk', index=False)

