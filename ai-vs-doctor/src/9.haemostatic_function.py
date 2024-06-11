import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'CHARTTIME', 'ITEM', 'VALUE', 'UNIT', 'CURRENT_OR_HISTORICAL']

def extract_unit(value_unit):
    unit_list = ['sec','g/L','ug/ml']
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
        next_check_list = ast.literal_eval(row['下一步检查（当前）'])
        for next_check in next_check_list:
            if list(next_check.keys())[0] == '止凝血':
                value_list_dict = next_check.get('止凝血')
                time = value_list_dict.get('时间')
                value_list = value_list_dict.get('值')
                for value in value_list:
                    item = list(value.keys())[0]
                    v,u = extract_unit(value.get(item))
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(time)
                    row_data.append(item)
                    row_data.append(v)
                    row_data.append(u)
                    row_data.append(CUR)
                    df.loc[len(df)] = row_data
        check_his_dict = row['下一步检查_止凝血（历史）']
        if str(check_his_dict) != 'nan':
            check_his_dict = eval(check_his_dict)
            value_lists = check_his_dict.get(list(check_his_dict.keys())[0])
            for value_dict in value_lists:
                time = value_dict.get('时间')
                value_list = value_dict.get('值')
                for value in value_list:
                    item = list(value.keys())[0]
                    v, u = extract_unit(value.get(item))
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(time)
                    row_data.append(item)
                    row_data.append(v)
                    row_data.append(u)
                    row_data.append(HIS)
                    df.loc[len(df)] = row_data
    df.to_csv(root + '\\datasets\\Haemostatic_Function.csv', encoding='gbk', index=False)


