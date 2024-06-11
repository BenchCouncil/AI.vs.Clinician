import numpy as np
import pandas as pd



def get_mimiciv_train():
    df_ill = pd.read_csv(
        'D:\\4-work\\14-mimic-iv\\3-sample-select\\0-mimiciv_ill_redio_subject_data_distribute_from_admittime.csv',
        encoding='gbk')
    # 这里不用患病患者3h的了，会导致0h模型中样本的不平衡，就选那些0h和-3h
    # df_ill = df_ill[df_ill['time_range'] != '3h']

    df_not = pd.read_csv('D:\\4-work\\14-mimic-iv\\3-sample-select\\1-mimiciv_not_redio_subject_data_distribute.csv',
                         encoding='gbk')

    df_sample = pd.read_csv('D:\\4-work\\14-mimic-iv\\3-sample-select\\10-mimiciv_3000_sample.csv', encoding='gbk')

    df_not['starttime'] = df_not['start_endtime'].str.split('~').str[0]
    df_not['endtime'] = df_not['start_endtime'].str.split('~').str[1]

    df = pd.concat([df_ill, df_not])

    df_train = df[~df['subject_id'].isin(df_sample['subject_id'])]

    df_train = df_train.drop_duplicates(subset=['hadm_id', 'time_range'])

    df_train = df_train.sample(frac=1, random_state=42)
    print(len(df_train))
    df_train = df_train[
        ['subject_id', 'hadm_id', 'gender', 'admittime', 'start_endtime', 'ill_time', 'time_range', 'starttime', 'endtime']]
    df_train.drop_duplicates(subset=['hadm_id']).to_csv(
        'D:\\4-work\\14-mimic-iv\\3-sample-select\\13-mimiciv_train_5_4_from_admittime.csv', encoding='gbk',
        index=False)



if __name__ == "__main__":

    df = pd.read_csv('D:\\4-work\\14-mimic-iv\\3-sample-select\\13-mimiciv_train_5_4_from_admittime.csv',encoding='gbk')

    df_ill_subject = df[~df['ill_time'].isna()]
    df_not_subject = df[df['ill_time'].isna()]

    df_ill_subject_2 = df_ill_subject.sample(int(len(df_ill_subject) * 0.2))
    df_ill_subject_remain = df_ill_subject[~df_ill_subject['hadm_id'].isin(df_ill_subject_2['hadm_id'])]
    df_ill_subject_1 = df_ill_subject_remain.sample(int(len(df_ill_subject) * 0.1))
    df_ill_subject_7 = df_ill_subject[
        ~df_ill_subject['hadm_id'].isin(set(df_ill_subject_1['hadm_id']) | set(df_ill_subject_2['hadm_id']))]

    df_not_subject_2 = df_not_subject.sample(int(len(df_not_subject) * 0.2))
    df_not_subject_remain = df_not_subject[~df_not_subject['hadm_id'].isin(df_not_subject_2['hadm_id'])]
    df_not_subject_1 = df_not_subject_remain.sample(int(len(df_not_subject) * 0.1))
    df_not_subject_7 = df_not_subject[
        ~df_not_subject['hadm_id'].isin(set(df_not_subject_1['hadm_id']) | set(df_not_subject_2['hadm_id']))]

    df_ill_1 = df_ill_subject[df_ill_subject['hadm_id'].isin(set(df_ill_subject_1['hadm_id']))]
    df_ill_2 = df_ill_subject[df_ill_subject['hadm_id'].isin(set(df_ill_subject_2['hadm_id']))]
    df_ill_7 = df_ill_subject[df_ill_subject['hadm_id'].isin(set(df_ill_subject_7['hadm_id']))]
    df_not_7 = df_not_subject[df_not_subject['hadm_id'].isin(set(df_not_subject_7['hadm_id']))]
    df_not_2 = df_not_subject[df_not_subject['hadm_id'].isin(set(df_not_subject_2['hadm_id']))]
    df_not_1 = df_not_subject[df_not_subject['hadm_id'].isin(set(df_not_subject_1['hadm_id']))]

    df_train = pd.concat([df_ill_7, df_not_7])
    df_val = pd.concat([df_ill_1, df_not_1])
    df_test = pd.concat([df_ill_2, df_not_2])

    df_train[['hadm_id']].drop_duplicates().to_csv(
        'D:\\4-work\\14-mimic-iv\\3-sample-select\\13-mimiciv_lstm_train_hadm_id.csv', index=False, mode='w')
    df_val[['hadm_id']].drop_duplicates().to_csv(
        'D:\\4-work\\14-mimic-iv\\3-sample-select\\13-mimiciv_lstm_val_hadm_id.csv', index=False, mode='w')
    df_test[['hadm_id']].drop_duplicates().to_csv(
        'D:\\4-work\\14-mimic-iv\\3-sample-select\\13-mimiciv_lstm_test_hadm_id.csv', index=False, mode='w')
    print(len(df_train[['hadm_id']].drop_duplicates()))
    print(len(df_val[['hadm_id']].drop_duplicates()))
    print(len(df_test[['hadm_id']].drop_duplicates()))







