import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'INPUTTIME', 'INPUT_AMOUNT', 'MEDICATION_RATE', 'FLUID_NAME',  'FLUID_TYPE']

def extract_unit(value_unit):
    unit_list = ['°C','mmHg','bpm','insp/min']
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
        df_case_info_id = df_case_info[df_case_info['CURRENT_TIME'] == row['START_ENDTIME']]
        case_id = df_case_info_id.iloc[0]['PATIENT_CASE_ID']

        input_list_dict = row['基础信息_输入（历史）']
        print(str(input_list_dict))
        if str(input_list_dict) != 'nan':
            input_list_dict = eval(input_list_dict)
            item = list(input_list_dict.keys())[0]
            input_list = input_list_dict.get(item)
            for value_dict in input_list:
                input_time = value_dict.get('时间')
                value_list = value_dict.get('值')
                for value in value_list:
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(input_time)
                    row_data.append(value.get('入量'))
                    row_data.append(value.get('给药速率'))
                    temp = value.get('补液名称')
                    v_after = temp[temp.find('|')+1:]
                    row_data.append(v_after)
                    row_data.append(value.get('补液类型'))
                    df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\FLUID_INPUT.csv', encoding='gbk', index=False)