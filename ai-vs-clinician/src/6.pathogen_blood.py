import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'CHARTTIME', 'ITEM', 'BODY_PART', 'VALUE', 'UNIT', 'CURRENT_OR_HISTORICAL']


if __name__ == '__main__':
    df_sample_all = pd.read_csv(fn_sample_all, encoding='gbk')
    df_sample = df_sample_all.drop_duplicates(subset=['START_ENDTIME'])

    df_case_info = pd.read_csv(root + 'datasets\\Patient_Case_Info.csv', encoding='gbk')
    df = pd.DataFrame(columns=columns, dtype='object')

    for index, row in df_sample.iterrows():
        df_case_info_id = df_case_info[df_case_info['CURRENT_TIME'] == row['START_ENDTIME']]
        case_id = df_case_info_id.iloc[0]['PATIENT_CASE_ID']
        next_check_list = ast.literal_eval(row['下一步检查（当前）'])
        for next_check in next_check_list:
            item = list(next_check.keys())[0]
            if item == '病原血检查':
                value_list_dict = next_check.get('病原血检查')
                time = value_list_dict.get('时间')
                value = value_list_dict.get('值')
                part = value_list_dict.get('病原血部位类别')
                row_data = []
                row_data.append(case_id)
                row_data.append(time)
                row_data.append(item)
                row_data.append(part)
                row_data.append(value)
                row_data.append(None)
                row_data.append(CUR)
                df.loc[len(df)] = row_data
        check_his_dict = row['下一步检查_病原血检查（历史）']
        if str(check_his_dict) != 'nan':
            check_his_dict = eval(check_his_dict)
            item = list(check_his_dict.keys())[0]
            value_lists = check_his_dict.get(item)
            for value_dict in value_lists:
                time = value_dict.get('时间')
                value = value_dict.get('值')
                part = value_dict.get('病原血部位类别')
                row_data = []
                row_data.append(case_id)
                row_data.append(time)
                row_data.append(item)
                row_data.append(part)
                row_data.append(value)
                row_data.append(None)
                row_data.append(HIS)
                df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Pathogen_Blood.csv', encoding='gbk', index=False)


