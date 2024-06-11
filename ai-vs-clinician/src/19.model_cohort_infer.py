import pandas as pd
from src.constant import *
import ast

columns = ['MODEL_INFER_ID', 'MODEL_ID', 'PATIENT_CASE_ID', 'PREDICTION_RESULT', 'PROBABILITY_0H',  'PROBABILITY_3H']


if __name__ == '__main__':
    df_patient_case = pd.read_csv(root + 'datasets\\Patient_Case_Info.csv', encoding='gbk')
    df_patient_case['END_TIME'] = df_patient_case['CURRENT_TIME'].str.split('~').str[1]
    df_patient_case['END_TIME'] = pd.to_datetime(df_patient_case['END_TIME'])
    df_model = pd.read_csv(root+'datasets\\Model_Property.csv',encoding='gbk')


    df = pd.DataFrame(columns=columns, dtype='object')

    df_group1_lstm75 = pd.read_csv(root+'data\\model_infer\\lstm_mimiciv_3000_sample_predict_result_group1_75.csv',encoding='gbk')
    df_group1_lstm85 = pd.read_csv(root+'data\\model_infer\\lstm_mimiciv_3000_sample_predict_result_group1_85.csv',encoding='gbk')
    df_group1_lstm95 = pd.read_csv(root+'data\\model_infer\\lstm_mimiciv_3000_sample_predict_result_group1_95.csv',encoding='gbk')
    df_group1_random = pd.read_csv(root+'data\\model_infer\\lstm_mimiciv_3000_sample_predict_result_group1_random.csv',encoding='gbk')
    df_group3_lstm95 = pd.read_csv(root+'data\\model_infer\\lstm_mimiciv_3000_sample_predict_result_group3_95.csv',encoding='gbk')
    df_group4_lstm85 = pd.read_csv(root+'data\\model_infer\\lstm_mimiciv_3000_sample_predict_result_group4_85.csv',encoding='gbk')
    df_group5_lstm75 = pd.read_csv(root+'data\\model_infer\\lstm_mimiciv_3000_sample_predict_result_group5_75.csv',encoding='gbk')
    df_group1_tscore = pd.read_csv(root+'data\\model_infer\\tscore_predict_result_no12h_group1.0.csv')
    df_group2_tscore = pd.read_csv(root+'data\\model_infer\\tscore_predict_result_no12h_group2.0.csv')

    model_infer_id = 1
    for model_name,df_lstm in [('Low_LSTM',df_group1_lstm75),('Medium_LSTM',df_group1_lstm85),('High_LSTM',df_group1_lstm95),
                    ('RandomModel',df_group1_random),('High_LSTM',df_group3_lstm95),
                    ('Medium_LSTM',df_group4_lstm85),('Low_LSTM',df_group5_lstm75),
                    ('CoxPHM', df_group1_tscore), ('CoxPHM', df_group2_tscore)]:

        for index,row in df_lstm.iterrows():
            df_model_id = df_model[df_model['MODEL_NAME'] == model_name]
            model_id = df_model_id.iloc[0]['MODEL_ID']

            df_patient = df_patient_case[df_patient_case['END_TIME'] == pd.to_datetime(row['endtime'])]
            patient_case_id = df_patient.iloc[0]['PATIENT_CASE_ID']

            predict_result = row['predict_result']
            if 'TREWScore' in predict_result:
                predict_result = predict_result.replace(":", ":'", 1).replace(",", "',", 1)
                predict_result = predict_result.replace("No", "'No'")
                predict_result = predict_result.replace("Yes", "'Yes'")
            predict_result = ast.literal_eval(predict_result)
            # print(predict_result)

            predict_pro_0h = None
            predict_pro_3h = None
            predict_sepsis = None

            for key in list(predict_result.keys()):
                if 'IsVisible' in key:
                    continue
                elif 'Predict_Time' in key:
                    predict_pro = predict_result.get(key)
                    predict_pro = predict_pro.split(',')
                    predict_pro_0h = predict_pro[0].split(':')[1]
                    predict_pro_3h = predict_pro[1].split(':')[1]
                else:
                    predict_sepsis = predict_result.get(key)

            row_data = []
            row_data.append(model_infer_id)
            row_data.append(model_id)
            row_data.append(patient_case_id)
            row_data.append(predict_sepsis)
            row_data.append(predict_pro_0h)
            row_data.append(predict_pro_3h)
            model_infer_id+=1
            df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Model_Cohort_Infer.csv', encoding='gbk', index=False)
