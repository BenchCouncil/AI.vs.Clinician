import  pandas as pd
import random
#22节点上

PATH = '/home/ddcui/'
ROOTIII = f'{PATH}/hai-med-database/mimic-coxphm/data/mimiciii_preprocess/'
ROOTIV = f'{PATH}/hai-med-database/mimic-coxphm/data/mimiciv_preprocess/'

def del_column(df):
    del df['subject_id']
    del df['hadm_id']
    del df['admittime']
    if 'illtime' in df.columns:
        del df['illtime']
    del df['starttime']
    del df['endtime']
    return df

def connect():
    #mimic iii
    df_iii_ill = pd.read_csv(ROOTIII + 'tscore_mimiciii_ill_model_input_data_by_diff_meanvalue.csv')
    df_iii_not = pd.read_csv(ROOTIII + 'tscore_mimiciii_not_model_input_data_by_diff_meanvalue.csv')
    df_iii = pd.concat([df_iii_not,df_iii_ill])

    df_iii_train_id = pd.read_csv(ROOTIII + '06-lstm_train_hadm_id.csv')
    df_iii_val_id = pd.read_csv(ROOTIII + '07-lstm_val_hadm_id.csv')
    df_iii_test_id = pd.read_csv(ROOTIII + '08-lstm_test_hadm_id.csv')

    df_iii_train = df_iii[df_iii['hadm_id'].isin(set(df_iii_train_id['hadm_id']) | set(df_iii_val_id['hadm_id']))]
    df_iii_test = df_iii[df_iii['hadm_id'].isin(set(df_iii_test_id['hadm_id']))]

    #mimic iv

    df_iv_ill = pd.read_csv(ROOTIV + 'tscore_mimiciv_ill_model_input_data_by_diff_meanvalue.csv')
    df_iv_not = pd.read_csv(ROOTIV + 'tscore_mimiciv_not_model_input_data_by_diff_meanvalue.csv')
    df_iv = pd.concat([df_iv_not,df_iv_ill])

    df_iv_train_id = pd.read_csv(ROOTIV + '13-mimiciv_lstm_train_hadm_id.csv')
    df_iv_val_id = pd.read_csv(ROOTIV + '13-mimiciv_lstm_val_hadm_id.csv')
    df_iv_test_id = pd.read_csv(ROOTIV + '13-mimiciv_lstm_test_hadm_id.csv')

    df_iv_train = df_iv[df_iv['hadm_id'].isin(set(df_iv_train_id['hadm_id']) | set(df_iv_val_id['hadm_id']))]
    df_iv_test = df_iv[df_iv['hadm_id'].isin(set(df_iv_test_id['hadm_id']))]

    df_train = pd.concat([df_iii_train,df_iv_train])
    df_test = pd.concat([df_iii_test,df_iv_test])

    df_train = del_column(df_train)
    df_test = del_column(df_test)

    PATH = '/home/ddcui/'
    TO_ROOT = f'{PATH}/hai-med-database/mimic-coxphm/data/connect_data/'

    df_train.to_csv(TO_ROOT+'mimiciv_and_mimiciii_lossrate_40_train_80_tscore_model_input_data.csv', index=False)
    df_test.to_csv(TO_ROOT+'mimiciv_and_mimiciii_lossrate_40_test_20_tscore_model_input_data.csv', index=False)

if __name__ == '__main__':
    connect()
    print('end')
