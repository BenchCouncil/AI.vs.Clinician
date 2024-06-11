import pandas as pd
from src.constant import *


if __name__ == '__main__':

    df_lstm = pd.read_csv(root+'\\data\\model_feature\\lstm_mimiciv_sample_model_input_data_by_diff_meanvalue_predict.csv')
    df_lstm.columns = df_lstm.columns.str.upper()
    df_lstm = df_lstm[df_lstm['HADM_ID'] != -4]

    df_patient_case = pd.read_csv(root+'\\datasets\\Patient_Case_Info.csv',encoding='gbk')
    df_patient_case['START_TIME'] = df_patient_case['CURRENT_TIME'].str.split('~').str[0]
    df_patient_case['START_TIME'] = pd.to_datetime(df_patient_case['START_TIME'])

    for index, row in df_lstm.iterrows():
        row_data = []
        df_patient = df_patient_case[df_patient_case['START_TIME'] == pd.to_datetime(row['START_TIME'])]
        patient_case_id = df_patient.iloc[0]['PATIENT_CASE_ID']

        df_lstm.at[index, 'PATIENT_CASE_ID'] = patient_case_id

    del df_lstm['HADM_ID']
    del df_lstm['START_TIME']
    del df_lstm['END_TIME']

    first_column = df_lstm.pop('PATIENT_CASE_ID')
    df_lstm.insert(0, 'PATIENT_CASE_ID', first_column)
    df_lstm.to_csv(root + '\\datasets\\LSTM_Feature.csv', encoding='gbk', index=False)