import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'CHARTTIME', 'ITEM', 'VALUE', 'UNIT']

def extract_unit(value_unit):
    unit_list = ['mmHg','umol/L',' 10^9/L','umol/L']
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


        sofa1 = eval(row['SOFA_呼吸系统'])
        sofa2 = eval(row['SOFA_凝血系统'])
        sofa3 = eval(row['SOFA_肝脏'])
        sofa4 = eval(row['SOFA_心血管系统'])
        sofa5 = eval(row['SOFA_中枢神经系统'])
        sofa6 = eval(row['SOFA_肾脏'])

        for sofa_list in [['SOFA_呼吸系统',sofa1],['SOFA_凝血系统',sofa2],['SOFA_肝脏',sofa3],['SOFA_心血管系统',sofa4],['SOFA_中枢神经系统',sofa5],['SOFA_肾脏',sofa6]]:
            item_str, sofa = sofa_list[0],sofa_list[1]
            for item in list(sofa.keys()):
                if item != '所属类别':
                    value_unit = sofa.get(item)
                    v,u = extract_unit(value_unit)
                    item_after = f'{item_str}_{item}'
                    row_data = []
                    row_data.append(case_id)
                    row_data.append(time)
                    row_data.append(item_after)
                    row_data.append(v)
                    row_data.append(u)
                    df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\SOFA.csv', encoding='gbk', index=False)


