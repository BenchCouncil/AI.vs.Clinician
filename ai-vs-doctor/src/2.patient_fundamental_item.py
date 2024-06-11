import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'CHARTTIME', 'BASE_ITEM', 'VALUE', 'UNIT', 'CURRENT_OR_HISTORICAL']

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

    df_case_info = pd.read_csv(root+'datasets\\Patient_Case_Info.csv',encoding='gbk')
    df = pd.DataFrame(columns=columns,dtype='object')

    for index,row in df_sample.iterrows():
        df_case_info_id = df_case_info[df_case_info['CURRENT_TIME'] == row['START_ENDTIME']]
        case_id = df_case_info_id.iloc[0]['PATIENT_CASE_ID']
        fun_currs = row['基础信息（当前）']
        fun_curr_list = ast.literal_eval(fun_currs)

        qsofa_blood = None
        #当前信息
        for fun_curr_dict in fun_curr_list:
            item = list(fun_curr_dict.keys())[0]
            if item != '体液24h' and item != 'QSOFA' and item != '血压':
                value_dict = fun_curr_dict.get(item)
                v,u = extract_unit(value_dict.get('值'))
                row_data = []
                row_data.append(case_id)
                row_data.append(value_dict.get('时间'))
                row_data.append(item)
                row_data.append(v)
                row_data.append(u)
                row_data.append(CUR)
                df.loc[len(df)] = row_data

            if item == 'QSOFA':
                value_dict = fun_curr_dict.get(item)
                for i in ['呼吸频率','收缩压','意识','QSOFA分数']:
                    v, u = extract_unit(value_dict.get(i))
                    if i == '收缩压':
                        qsofa_blood = v
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(value_dict.get('时间'))
                    row_data.append(f'{item}_{i}')
                    row_data.append(v)
                    row_data.append(u)
                    row_data.append(CUR)
                    df.loc[len(df)] = row_data

            if item == '体液24h':
                input_output = fun_curr_dict.get(item)
                input_output = input_output.replace('ml','')
                input_outputs = input_output.split('/')
                for item_temp,item_value in [('24-hour fluid input',input_outputs[0]),('24-hour fluid output',input_outputs[1])]:
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(None)
                    row_data.append(item_temp)
                    row_data.append(item_value)
                    row_data.append('ml')
                    row_data.append(None)
                    df.loc[len(df)] = row_data


        for fun_curr_dict in fun_curr_list:#单独处理血压
            item = list(fun_curr_dict.keys())[0]
            if item == '血压':
                value_dict = fun_curr_dict.get(item)
                v, u = extract_unit(value_dict.get('值'))
                row_data = []
                row_data.append(case_id)
                row_data.append(value_dict.get('时间'))
                row_data.append('收缩压/舒张压')
                v_after = v[v.find('/'):]
                row_data.append(f'{qsofa_blood}{v_after}')
                row_data.append(u)
                row_data.append(CUR)
                df.loc[len(df)] = row_data
                break

        #历史信息
        for check_his_dict in [row['基础信息_体温（历史）'],row['基础信息_血压（历史）'],row['基础信息_心率（历史）']]:
            print(str(check_his_dict))
            if str(check_his_dict) != 'nan':
                check_his_dict = eval(check_his_dict)
                item = list(check_his_dict.keys())[0]
                value_list = check_his_dict.get(item)
                for value_dict in value_list:
                    v, u = extract_unit(value_dict.get('值'))
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(value_dict.get('时间'))
                    if item == '血压':
                        item = '收缩压/舒张压'
                    row_data.append(item)
                    row_data.append(v)
                    row_data.append(u)
                    row_data.append(HIS)
                    df.loc[len(df)] = row_data
        qsofa_hiss = row['基础信息_QSOFA（历史）']
        if str(qsofa_hiss) != 'nan':
            qsofa_hiss = ast.literal_eval(qsofa_hiss)
            for qsofa_his_dict in qsofa_hiss:
                item = list(qsofa_his_dict.keys())[0]
                if item == '呼吸频率':
                    value_list = qsofa_his_dict.get(item)
                    for value_dict in value_list:
                        v, u = extract_unit(value_dict.get('值'))
                        row_data = []
                        row_data.append(case_id)
                        row_data.append(value_dict.get('时间'))
                        row_data.append(item)
                        row_data.append(v)
                        row_data.append(u)
                        row_data.append(HIS)
                        df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Patient_Fundamental_ITEM.csv', encoding='gbk', index=False)


