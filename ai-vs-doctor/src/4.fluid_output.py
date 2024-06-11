import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID',  'OUTPUTTIME',  'OUTPUT_AMOUNT',  'FLUID_NAME']

if __name__ == '__main__':
    df_sample_all = pd.read_csv(fn_sample_all, encoding='gbk')
    df_sample = df_sample_all.drop_duplicates(subset=['START_ENDTIME'])

    df_case_info = pd.read_csv(root + 'datasets\\Patient_Case_Info.csv', encoding='gbk')
    df = pd.DataFrame(columns=columns, dtype='object')

    for index, row in df_sample.iterrows():
        df_case_info_id = df_case_info[df_case_info['CURRENT_TIME'] == row['START_ENDTIME']]
        case_id = df_case_info_id.iloc[0]['PATIENT_CASE_ID']

        input_list_dict = row['基础信息_输出（历史）']
        print(str(input_list_dict))
        if str(input_list_dict) != 'nan':
            input_list_dict = eval(input_list_dict)
            item = list(input_list_dict.keys())[0]
            input_list = input_list_dict.get(item)
            for value_dict in input_list:
                output_time = value_dict.get('时间')
                value_list = value_dict.get('值')
                for value in value_list:
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(output_time)
                    row_data.append(value.get('出量'))
                    temp = value.get('出液名称')
                    v_after = temp[temp.find('|') + 1:]
                    row_data.append(v_after)
                    df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\FLUID_OUTPUT.csv', encoding='gbk', index=False)