import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'CHARTTIME', 'ITEM', 'VALUE', 'UNIT']

def extract_unit(value_unit):
    unit_list = ['ng/ml']
    for u in unit_list:
        if u in str(value_unit):
            value = value_unit.replace(u,'')
            unit = u
            return value, unit
    return value_unit,None


if __name__ == '__main__':
    df_sample_all = pd.read_csv(fn_sample_all, encoding='gbk')
    df_sample = df_sample_all.drop_duplicates(subset=['START_ENDTIME'])

    df_case_info = pd.read_csv(root + 'datasets\\Patient_Case_Info.csv', encoding='gbk')
    df = pd.DataFrame(columns=columns, dtype='object')

    for index, row in df_sample.iterrows():
        print(index)
        df_case_info_id = df_case_info[df_case_info['CURRENT_TIME'] == row['START_ENDTIME']]
        case_id = df_case_info_id.iloc[0]['PATIENT_CASE_ID']

        fun_currs = row['基础信息（当前）']
        fun_curr_list = ast.literal_eval(fun_currs)
        item_temp = list(fun_curr_list[0].keys())[0]
        value_dict = fun_curr_list[0].get(item_temp)
        time = value_dict.get('时间')


        value_unit = row['降钙素原']
        v,u = extract_unit(value_unit)
        row_data = []
        row_data.append(case_id)
        row_data.append(time)
        row_data.append('降钙素原')
        row_data.append(v)
        row_data.append(u)
        df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Procalcitonin.csv', encoding='gbk', index=False)


