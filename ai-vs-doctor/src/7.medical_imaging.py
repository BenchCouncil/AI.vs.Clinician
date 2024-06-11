import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'CHARTTIME', 'ITEM',  'IMAGE_TYPE', 'IMAGE_BODYPART', 'IMAGE_JPG', 'TEXT_REPORT', 'CURRENT_OR_HISTORICAL']

def get_report_en(text):
    last_colon_index = text.rfind("英文报告：")  # 找到最后一个冒号的索引位置
    if last_colon_index != -1:  # 如果找到了冒号
        return text[last_colon_index+5:]
    return text


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
            if item == '影像报告':
                value_lists = next_check.get('影像报告')
                for value_dict in value_lists:
                    time = value_dict.get('时间')
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(time)
                    row_data.append(item)
                    row_data.append(value_dict.get('影像类型'))
                    row_data.append(value_dict.get('影像部位'))
                    row_data.append(value_dict.get('影像图片'))
                    row_data.append(get_report_en(value_dict.get('值')))
                    row_data.append(CUR)
                    df.loc[len(df)] = row_data
        check_his_dict = row['下一步检查_影像报告（历史）']
        if str(check_his_dict) != 'nan':
            check_his_dict = eval(check_his_dict)
            item = list(check_his_dict.keys())[0]
            value_lists = check_his_dict.get(item)
            for value_dict in value_lists:
                time = value_dict.get('时间')
                value_list = value_dict.get('值')
                row_data = []
                row_data.append(case_id)
                row_data.append(time)
                row_data.append(item)
                row_data.append(value_dict.get('影像类型'))
                row_data.append(value_dict.get('影像部位'))
                row_data.append(value_dict.get('影像图片'))
                row_data.append(get_report_en(value_dict.get('值')))
                row_data.append(HIS)
                df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Medical_Imaging.csv', encoding='gbk', index=False)
