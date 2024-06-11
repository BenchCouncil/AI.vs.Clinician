# coding=gbk
import pandas as pd
import copy
import ast
import csv


# �����ʵ�λ�޸�Ϊ����ͨ�õ�λ������ҽ�����

def base_current(list_str):
    list_str = eval(str(list_str))
    list_str_after = []
    for dict_str in list_str:
        dict_str_af = dict_str
        if 'Ѫ����' in dict_str.keys():
            dict_str_af = chart_current(dict_str, 'Ѫ����')
        if '����Ѫ������' in dict_str.keys():
            dict_str_af = chart_current(dict_str, '����Ѫ������')
        if 'ֹ��Ѫ' in dict_str.keys():
            dict_str_af = chart_current(dict_str, 'ֹ��Ѫ')
        list_str_after.append(dict_str_af)
    return list_str_after

def chart_current(dict_str, kind):
    dict_br = ast.literal_eval(str(dict_str))
    list_br = dict_br[kind]
    time_list = list_br['ֵ']
    time_list_after = []
    for time_dict in time_list:
        if time_dict :
            key, value = time_dict.popitem()
            value_after = None
            if kind == 'Ѫ����':
                value_after = blood_routine(key, value)
            elif kind == '����Ѫ������':
                value_after = gas_anyle(key, value)
            elif kind == 'ֹ��Ѫ':
                value_after = stop_blood(key, value)
            if value_after is not None:
                time_dict[key] = value_after
                time_list_after.append(time_dict)
    list_br['ֵ'] = time_list_after
    return dict_br

def chart_history(dict_str, kind):
    if dict_str == 'nan':
        return None
    dict_br = ast.literal_eval(str(dict_str))
    list_br = dict_br[kind]
    for time_br in list_br:
        time_list = time_br['ֵ']
        time_list_after = []
        for time_dict in time_list:
            if time_dict:
                key, value = time_dict.popitem()
                value_after = None
                if kind == 'Ѫ����':
                    value_after = blood_routine(key, value)
                elif kind == '����Ѫ������':
                    value_after = gas_anyle(key, value)
                elif kind == 'ֹ��Ѫ':
                    value_after = stop_blood(key, value)
                if value_after is not None:
                    time_dict[key] = value_after
                    time_list_after.append(time_dict)
        time_br['ֵ'] = time_list_after
        print(time_list_after)
    return dict_br

#����Ѫ�����еļ���λ
def blood_routine(key, value):
    value = str(value).lower()
    if value == '0%':
        return None
    if key == 'WBC|��ϸ������' or key == 'PLT|ѪС�����' or key == 'LYMPH#|�ܰ�ϸ������ֵ':
        if 'k/ul' in value:
            value = value.replace('k/ul', ' 10^9/L')
        else:
            return None
    elif key == 'HGB|Ѫ�쵰��Ũ��' or key == 'MCHC|ƽ��Ѫ�쵰��Ũ��' or key == 'Albumin|�׵���':
        if 'g/dl' in value:
            value = float(value.replace('g/dl', '')) * 10
            value = str(value) + 'g/L'
        else:
            return None
    elif key == 'BUN|���ص�':
        if 'mg/dl' in value:
            value = float(value.replace('mg/dl', '')) * 0.365
            value = str(round(value, 2)) + 'mmol/L'
        else:
            return None
    elif key == 'Cr|����':
        if 'mg/dl' in value:
            value = float(value.replace('mg/dl', '')) * 88.4
            value = str(round(value, 2)) + 'umol/L'
        else:
            return None
    return value

#����Ѫ�������еļ���λ
def gas_anyle(key, value):
    value = str(value).lower()
    if value == '0%':
        return None
    if key == 'BE|ʣ���' or key == 'TCO2|�ܶ�����̼' or key == 'CL|������' or key == 'K+|��' or key == 'Sodium Urine|��Һ�е���' or key == 'Sodium Whole Blood|ȫѪ�е���' or key == 'HCO3|̼�����(Ѫ��)':
        if 'meq/l' in value:
            value = value.replace('meq/l', 'mmol/L')
        else:
            return None
    elif key == 'GLU|������':
        if 'mg/dl' in value:
            if 'negmg/dl' == value:
                return None
            elif '>1000mg/dl' == value:
                return '>55mmol/L'
            value = float(value.replace('mg/dl', '')) / 18
            value = str(round(value, 2)) + 'mmol/L'
        else:
            return None
    elif key == 'm0sm|��͸ѹ':
        if 'mOsm/kg' in value:
            value = value.replace('mOsm/kg', 'mOsm/L')
        else:
            return None
    return value

#����ֹ��Ѫ�еļ���λ
def stop_blood(key, value):
    value = str(value).lower()
    if key == 'FIB|��ά����ԭ':
        if 'mg/dl' in value:
            value = float(value.replace('mg/dl', '')) / 100
            value = str(round(value, 2)) + 'g/L'
        else:
            return None
    elif key == 'D-D2(||)|D-�����壨II��':
        if 'ng/ml' in value:
            value = value.replace(' feu', '')
            value = float(value.replace('ng/ml', '')) / 1000
            value = str(round(value, 2)) + 'ug/ml'
        else:
            return None
    return value

#ͳһsofa�е�λ
def main(src_path,to_path):
    df = pd.read_csv(src_path, encoding='gbk')
    for index, row in df.iterrows():
        nc = str(row['��һ����飨��ǰ��'])
        br = str(row['��һ�����_Ѫ���棨��ʷ��'])
        ga = str(row['��һ�����_����Ѫ����������ʷ��'])
        sb = str(row['��һ�����_ֹ��Ѫ����ʷ��'])

        nc_after = base_current(nc)
        br_after = chart_history(br, 'Ѫ����')
        ga_after = chart_history(ga, '����Ѫ������')
        sb_after = chart_history(sb, 'ֹ��Ѫ')

        df.at[index, '��һ����飨��ǰ��'] = str(nc_after)
        df.at[index, '��һ�����_Ѫ���棨��ʷ��'] = br_after
        df.at[index, '��һ�����_����Ѫ����������ʷ��'] = ga_after
        df.at[index, '��һ�����_ֹ��Ѫ����ʷ��'] = sb_after

        sofa_1 = str(row['SOFA_��Ѫϵͳ'])
        sofa_2 = str(row['SOFA_����'])
        sofa_3 = str(row['SOFA_����'])

        sofa_1 = ast.literal_eval(str(sofa_1))
        value = sofa_1['ѪС��']
        value = value.replace('K/uL', ' 10^9/L')
        sofa_1['ѪС��'] = value

        # ������ 1mg/dl��17.1��mol/L
        sofa_2 = ast.literal_eval(str(sofa_2))
        value = sofa_2['������']
        value = float(value.replace('mg/dl', '')) * 17.1
        value = str(round(value, 2)) + 'umol/L'
        sofa_2['������'] = value

        sofa_3 = ast.literal_eval(str(sofa_3))
        value = sofa_3['�� ��']
        value = float(value.replace('mg/dl', '')) * 88.4
        value = str(round(value, 2)) + 'umol/L'
        sofa_3['�� ��'] = value

        df.at[index, 'SOFA_��Ѫϵͳ'] = sofa_1
        df.at[index, 'SOFA_����'] = sofa_2
        df.at[index, 'SOFA_����'] = sofa_3

    df.to_csv(to_path, mode='w', index=False, quoting=csv.QUOTE_ALL, encoding='gbk')

PATH = '/home/ddcui/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'

if __name__ == '__main__':
    #����mimiciv�м��������ݵ�λ�͹���ͨ�õ�λ�Ĳ��죬��˸��ļ����ݹ���ҽԺ��ͨ�õ�λ��mimic�е�λ������ת����Ŀ���Ƿ���ҽ�����
    src_path = TO_ROOT + 'front_end/mimiciv_sample_3000_front_end_display_data.csv'
    to_path = TO_ROOT + 'front_end/mimiciv_sample_3000_front_end_display_data_2.0.csv'
    main(src_path,to_path)
    print('end')
