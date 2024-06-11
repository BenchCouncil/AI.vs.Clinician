import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'CHARTTIME', 'ITEM', 'SPECIMEN_TYPE', 'VALUE', 'UNIT', 'CURRENT_OR_HISTORICAL']


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
            item = list(next_check.keys())[0]
            if item == '培养' or item == '涂片':
                value_list = next_check.get(item)
                for value in value_list:
                    time = value.get('时间')
                    v = value.get('值')
                    part = value.get('送检样本')
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(time)
                    row_data.append(item)
                    row_data.append(part)
                    row_data.append(v)
                    row_data.append(None)
                    row_data.append(CUR)
                    df.loc[len(df)] = row_data

        for check_his_dict in [row['下一步检查_培养（历史）'],row['下一步检查_涂片（历史）']]:
            if str(check_his_dict) != 'nan':
                check_his_dict = eval(check_his_dict)
                item = list(check_his_dict.keys())[0]
                value_lists = check_his_dict.get(item)
                for value_dict in value_lists:
                    time = value_dict.get('时间')
                    value = value_dict.get('值')
                    part = value_dict.get('送检样本')
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(time)
                    row_data.append(item)
                    row_data.append(part)
                    row_data.append(value)
                    row_data.append(None)
                    row_data.append(HIS)
                    df.loc[len(df)] = row_data
    df.to_csv(root + '\\datasets\\Culture_Smear.csv', encoding='gbk', index=False)


