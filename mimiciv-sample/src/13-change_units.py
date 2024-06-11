# coding=gbk
import pandas as pd
import copy
import ast
import csv


# 将国际单位修改为国内通用单位，方便医生诊断

def base_current(list_str):
    list_str = eval(str(list_str))
    list_str_after = []
    for dict_str in list_str:
        dict_str_af = dict_str
        if '血常规' in dict_str.keys():
            dict_str_af = chart_current(dict_str, '血常规')
        if '动脉血气分析' in dict_str.keys():
            dict_str_af = chart_current(dict_str, '动脉血气分析')
        if '止凝血' in dict_str.keys():
            dict_str_af = chart_current(dict_str, '止凝血')
        list_str_after.append(dict_str_af)
    return list_str_after

def chart_current(dict_str, kind):
    dict_br = ast.literal_eval(str(dict_str))
    list_br = dict_br[kind]
    time_list = list_br['值']
    time_list_after = []
    for time_dict in time_list:
        if time_dict :
            key, value = time_dict.popitem()
            value_after = None
            if kind == '血常规':
                value_after = blood_routine(key, value)
            elif kind == '动脉血气分析':
                value_after = gas_anyle(key, value)
            elif kind == '止凝血':
                value_after = stop_blood(key, value)
            if value_after is not None:
                time_dict[key] = value_after
                time_list_after.append(time_dict)
    list_br['值'] = time_list_after
    return dict_br

def chart_history(dict_str, kind):
    if dict_str == 'nan':
        return None
    dict_br = ast.literal_eval(str(dict_str))
    list_br = dict_br[kind]
    for time_br in list_br:
        time_list = time_br['值']
        time_list_after = []
        for time_dict in time_list:
            if time_dict:
                key, value = time_dict.popitem()
                value_after = None
                if kind == '血常规':
                    value_after = blood_routine(key, value)
                elif kind == '动脉血气分析':
                    value_after = gas_anyle(key, value)
                elif kind == '止凝血':
                    value_after = stop_blood(key, value)
                if value_after is not None:
                    time_dict[key] = value_after
                    time_list_after.append(time_dict)
        time_br['值'] = time_list_after
        print(time_list_after)
    return dict_br

#换算血常规中的检查项单位
def blood_routine(key, value):
    value = str(value).lower()
    if value == '0%':
        return None
    if key == 'WBC|白细胞计数' or key == 'PLT|血小板计数' or key == 'LYMPH#|淋巴细胞绝对值':
        if 'k/ul' in value:
            value = value.replace('k/ul', ' 10^9/L')
        else:
            return None
    elif key == 'HGB|血红蛋白浓度' or key == 'MCHC|平均血红蛋白浓度' or key == 'Albumin|白蛋白':
        if 'g/dl' in value:
            value = float(value.replace('g/dl', '')) * 10
            value = str(value) + 'g/L'
        else:
            return None
    elif key == 'BUN|尿素氮':
        if 'mg/dl' in value:
            value = float(value.replace('mg/dl', '')) * 0.365
            value = str(round(value, 2)) + 'mmol/L'
        else:
            return None
    elif key == 'Cr|肌酐':
        if 'mg/dl' in value:
            value = float(value.replace('mg/dl', '')) * 88.4
            value = str(round(value, 2)) + 'umol/L'
        else:
            return None
    return value

#换算血气分析中的检查项单位
def gas_anyle(key, value):
    value = str(value).lower()
    if value == '0%':
        return None
    if key == 'BE|剩余碱' or key == 'TCO2|总二氧化碳' or key == 'CL|氯离子' or key == 'K+|钾' or key == 'Sodium Urine|尿液中的钠' or key == 'Sodium Whole Blood|全血中的钠' or key == 'HCO3|碳酸氢根(血清)':
        if 'meq/l' in value:
            value = value.replace('meq/l', 'mmol/L')
        else:
            return None
    elif key == 'GLU|葡萄糖':
        if 'mg/dl' in value:
            if 'negmg/dl' == value:
                return None
            elif '>1000mg/dl' == value:
                return '>55mmol/L'
            value = float(value.replace('mg/dl', '')) / 18
            value = str(round(value, 2)) + 'mmol/L'
        else:
            return None
    elif key == 'm0sm|渗透压':
        if 'mOsm/kg' in value:
            value = value.replace('mOsm/kg', 'mOsm/L')
        else:
            return None
    return value

#换算止凝血中的检查项单位
def stop_blood(key, value):
    value = str(value).lower()
    if key == 'FIB|纤维蛋白原':
        if 'mg/dl' in value:
            value = float(value.replace('mg/dl', '')) / 100
            value = str(round(value, 2)) + 'g/L'
        else:
            return None
    elif key == 'D-D2(||)|D-二聚体（II）':
        if 'ng/ml' in value:
            value = value.replace(' feu', '')
            value = float(value.replace('ng/ml', '')) / 1000
            value = str(round(value, 2)) + 'ug/ml'
        else:
            return None
    return value

#统一sofa中单位
def main(src_path,to_path):
    df = pd.read_csv(src_path, encoding='gbk')
    for index, row in df.iterrows():
        nc = str(row['下一步检查（当前）'])
        br = str(row['下一步检查_血常规（历史）'])
        ga = str(row['下一步检查_动脉血气分析（历史）'])
        sb = str(row['下一步检查_止凝血（历史）'])

        nc_after = base_current(nc)
        br_after = chart_history(br, '血常规')
        ga_after = chart_history(ga, '动脉血气分析')
        sb_after = chart_history(sb, '止凝血')

        df.at[index, '下一步检查（当前）'] = str(nc_after)
        df.at[index, '下一步检查_血常规（历史）'] = br_after
        df.at[index, '下一步检查_动脉血气分析（历史）'] = ga_after
        df.at[index, '下一步检查_止凝血（历史）'] = sb_after

        sofa_1 = str(row['SOFA_凝血系统'])
        sofa_2 = str(row['SOFA_肝脏'])
        sofa_3 = str(row['SOFA_肾脏'])

        sofa_1 = ast.literal_eval(str(sofa_1))
        value = sofa_1['血小板']
        value = value.replace('K/uL', ' 10^9/L')
        sofa_1['血小板'] = value

        # 胆红素 1mg/dl＝17.1μmol/L
        sofa_2 = ast.literal_eval(str(sofa_2))
        value = sofa_2['胆红素']
        value = float(value.replace('mg/dl', '')) * 17.1
        value = str(round(value, 2)) + 'umol/L'
        sofa_2['胆红素'] = value

        sofa_3 = ast.literal_eval(str(sofa_3))
        value = sofa_3['肌 酐']
        value = float(value.replace('mg/dl', '')) * 88.4
        value = str(round(value, 2)) + 'umol/L'
        sofa_3['肌 酐'] = value

        df.at[index, 'SOFA_凝血系统'] = sofa_1
        df.at[index, 'SOFA_肝脏'] = sofa_2
        df.at[index, 'SOFA_肾脏'] = sofa_3

    df.to_csv(to_path, mode='w', index=False, quoting=csv.QUOTE_ALL, encoding='gbk')

PATH = '/home/ddcui/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'

if __name__ == '__main__':
    #由于mimiciv中检查项的数据单位和国内通用单位的差异，因此该文件根据国内医院的通用单位将mimic中单位进行了转换，目的是方便医生诊断
    src_path = TO_ROOT + 'front_end/mimiciv_sample_3000_front_end_display_data.csv'
    to_path = TO_ROOT + 'front_end/mimiciv_sample_3000_front_end_display_data_2.0.csv'
    main(src_path,to_path)
    print('end')
