import numpy as np
import pandas as pd
import os
import uuid

PATH = '/home/ddcui/'
ROOT_MIMICIV = f'{PATH}mimic-original-data/mimiciv_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'

ma_status = {
    'MARRIED':'已婚',
    'WIDOWED':'丧偶',
    'DIVORCED':'离异',
    'SINGLE':'单身',
    'nan': '未知'
}

ethnicity = {
    'BLACK/AFRICAN AMERICAN':'黑人/非洲裔美国人',
    'WHITE':'白人',
    'HISPANIC/LATINO':'西班牙裔/拉丁裔',
    'AMERICAN INDIAN/ALASKA NATIVE':'美洲印第安人/阿拉斯加土著人',
    'OTHER':'其他',
    'UNKNOWN':'未知',
    'UNABLE TO OBTAIN':'无法获取',
    'ASIAN':'亚洲人',
    'nan': '未知'
}

def get_ethnicity():
    df_sample = pd.read_csv(TO_ROOT + 'preprocess/10-mimiciv_3000_sample.csv', encoding='gbk')

    df_admission = pd.read_csv(ROOT_MIMICIV + 'admissions.csv',
                               usecols=['hadm_id', 'dischtime', 'marital_status', 'ethnicity'])
    df_sample = pd.merge(df_sample, df_admission, on=['hadm_id'], how='inner')

    df_sample['marital_status'] = df_sample['marital_status'].fillna('未知')
    df_sample['ethnicity'] = df_sample['ethnicity'].fillna('未知')

    df_sample['marital_status'] = df_sample['marital_status'].map(ma_status)
    df_sample['ethnicity'] = df_sample['ethnicity'].map(ethnicity)

    df_sample = df_sample.rename(columns={'marital_status': '婚姻状态', 'ethnicity': '种族'})
    df_sample.to_csv(TO_ROOT + 'preprocess/10-mimiciv_3000_sample_add_ethnicity.csv', encoding='gbk',
                     index=False)


if __name__ == "__main__":
    get_ethnicity()