import pandas as pd
from src.constant import *

columns = ['MODEL_DATA_ID', 'SPLIT_TYPE', 'HADM_ID', 'MIMIC_SOURCE','SEPSIS_ONSET_TIME']


if __name__ == '__main__':
    df = pd.DataFrame(columns=columns, dtype='object')

    df_train_iii = pd.read_csv(root+'\\data\\hadm_id_add_illtime\\mimiciii-lstm_train_hadm_id_illtime.csv')
    df_val_iii = pd.read_csv(root+'\\data\\hadm_id_add_illtime\\mimiciii-lstm_val_hadm_id_illtime.csv')
    df_test_iii = pd.read_csv(root+'\\data\\hadm_id_add_illtime\\mimiciii-lstm_test_hadm_id_illtime.csv')

    df_train_iv = pd.read_csv(root+'\\data\\hadm_id_add_illtime\\mimiciv-lstm_train_hadm_id_illtime.csv')
    df_val_iv = pd.read_csv(root+'\\data\\hadm_id_add_illtime\\mimiciv-lstm_val_hadm_id_illtime.csv')
    df_test_iv = pd.read_csv(root+'\\data\\hadm_id_add_illtime\\mimiciv-lstm_test_hadm_id_illtime.csv')

    model_data_id = 1
    for mimic,split,df_data in [('mimiciii','train',df_train_iii),('mimiciii','val',df_val_iii),
                                ('mimiciii','test',df_test_iii),('mimiciv','train',df_train_iv),
                                ('mimiciv','val',df_val_iv),('mimiciv','test',df_test_iv)]:
        for index,row in df_data.iterrows():
            hadm_id = row['hadm_id']
            ill_time = row['ill_time']
            row_data = []
            row_data.append(model_data_id)
            row_data.append(split)
            row_data.append(hadm_id)
            row_data.append(mimic)
            row_data.append(ill_time)
            df.loc[len(df)] = row_data
            model_data_id+=1
    df.to_csv(root + '\\datasets\\Model_Dataset.csv', encoding='gbk', index=False)
