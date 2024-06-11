import  pandas as pd
import random
#22节点上


PATH = '/home/ddcui/'
ROOT = f'{PATH}/hai-med-database/mimic-coxphm/data/mimiciv_preprocess/'

def get_ill_period_end_to_illtime(illtime,end_time):
    period_end_to_illtime = int((illtime - end_time).total_seconds() / 3600)
    if period_end_to_illtime == 0:
        period_end_to_illtime = 0.01
    return period_end_to_illtime

def get_not_period_end_to_illtime():
    period_end_to_illtime = random.randint(336, 400)
    return period_end_to_illtime

def del_column(df):

    del df['admittime']
    del df['illtime']
    del df['current_period']

    df = df.rename(columns={'starttime':'start_time','endtime':'end_time','group':'group_id'})
    return df


if __name__ == '__main__':
    #先把数据弄成trscore的格式   然后根据D:\4-work\14-mimic-iv\3-sample-select\15-mimiciv_lstm_train_hadm_id.csv 分train val test
    df_mimiciv_sample = pd.read_csv(ROOT+'lstm_mimiciv_sample_model_input_data_by_diff_meanvalue.csv')

    df_sample_id = pd.read_csv(ROOT+'10-mimiciv_3000_sample.csv',usecols=['start_endtime','group'],encoding='gbk')
    df_sample_id = df_sample_id.rename(columns={'start_endtime':'current_period'})
    df_mimiciv_sample = pd.merge(df_mimiciv_sample,df_sample_id,on=['current_period'])

    print(len(df_mimiciv_sample.drop_duplicates(subset=['current_period'])))
    df_mimiciv_sample = df_mimiciv_sample.sort_values(by=['current_period', 'endtime'], ascending=[False, False])
    df_mimiciv_sample = df_mimiciv_sample.drop_duplicates(subset=['current_period'],keep='first')
    print(len(df_mimiciv_sample.drop_duplicates(subset=['current_period'])))
    print('==========================')

    df_mimiciv_sample = df_mimiciv_sample[(df_mimiciv_sample['group'] == 1) | (df_mimiciv_sample['group'] == 2)]
    df_mimiciv_sample = df_mimiciv_sample[(df_mimiciv_sample['group'] == 1) | (df_mimiciv_sample['group'] == 2)]

    df_not = df_mimiciv_sample[df_mimiciv_sample['illtime'].astype(str) == 'nan']
    df_ill = df_mimiciv_sample[df_mimiciv_sample['illtime'].astype(str) != 'nan']
    df_ill['ill_label'] = 1
    df_not['ill_label'] = 0

    df_ill['illtime'] = pd.to_datetime(df_ill['illtime'])
    df_ill['endtime'] = pd.to_datetime(df_ill['endtime'])
    df_ill['period_end_to_illtime'] = df_ill.apply(lambda row: get_ill_period_end_to_illtime(row['illtime'], row['endtime']), axis=1)

    df_not['period_end_to_illtime'] = df_not.apply(lambda row: get_not_period_end_to_illtime(), axis=1)

    print(len(df_ill))
    print(len(df_not))
    df = pd.concat([df_ill,df_not])

    df = del_column(df)
    PATH = '/home/ddcui/'
    TO_ROOT = f'{PATH}/hai-med-database/mimic-coxphm/data/connect_data/'
    df.to_csv(TO_ROOT + 'tscore_mimiciv_sample_3000_group_12_input_data.csv',index=False)
