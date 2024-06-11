import pandas as pd
from src.constant import *
from datetime import datetime

check_type_dict = {
    1: '血常规',
    2: '动脉血气分析',
    3: '止凝血',
    4: '影像检查',
    5: '病原检查',
    6: '培养',
    7: '涂片',
    8: '历史用药',
    9: '降钙素原'
}

def syslog_check_type(string):
    if str(string) != 'nan':
        parts = string.split('&')
        # 遍历每个部分，找到以 "checkType" 开头的部分
        for part in parts:
            if part.startswith('checkType'):
                # 从 "checkType" 后面的 ":" 后提取数字
                return int(part.split(':')[1])
    return None

columns = ['INTERACTION_ID', 'CLICKED_ITEM', 'CLICKED_TIME']


if __name__ == '__main__':
    df = pd.DataFrame(columns=columns, dtype='object')

    df_doctor_diag = pd.read_csv(root + 'datasets\\Clinician_Diagnosis_Treatment.csv',encoding='gbk')

    df_syslog = pd.read_csv(root + 'data\\doctor\\sys_log.csv')
    df_syslog = df_syslog[(df_syslog['patient_id'] <= 6000) | (df_syslog['patient_id'] > 20000)]
    df_syslog.loc[:, 'create_time'] = pd.to_datetime(df_syslog['create_time'],format='%Y/%m/%d %H:%M')
    df_syslog['check_type'] = df_syslog['exception'].apply(syslog_check_type)

    df_patient_check = pd.read_csv(root + 'data\\doctor\\patient_check.csv')
    df_patient_check = df_patient_check[(df_patient_check['patient_id'] <= 6000) | (df_patient_check['patient_id'] > 20000)]


    df_syslog_group = df_syslog.groupby(['accountname','patient_id'])
    df_patient_check_group = df_patient_check.groupby(['doctor_id','patient_id'])

    df_doctor_diag_group = df_doctor_diag.groupby(['INTERACTION_ID'])
    for (interraction_id,df_temp) in df_doctor_diag_group:
        print(interraction_id)
        doctor_logid = int(interraction_id.split('_')[0])
        patient_logid = int(interraction_id.split('_')[1])

        starttime = df_temp.iloc[0]['STARTTIME']
        endtime = df_temp.iloc[0]['ENDTIME']

        df_sys_log_id = df_syslog_group.get_group((doctor_logid,patient_logid))
        df_sys_log_id = df_sys_log_id[df_sys_log_id['module'] != '下一步检查'] #下一步检查去patient_check表中提取
        for index,row in df_sys_log_id.iterrows():
            creat_time = row['create_time'].timestamp()
            if creat_time < starttime or creat_time > endtime:
                continue
            item0 = row['module']
            item1 = check_type_dict.get(row['check_type'])
            if item1 is None:
                click_item = item0
            else:
                click_item = f'{item0}_{item1}'
            row_data = []
            row_data.append(interraction_id)
            row_data.append(click_item)
            row_data.append(creat_time)
            df.loc[len(df)] = row_data

        df_patient_check_id = None
        if (doctor_logid, patient_logid) in df_patient_check_group.groups.keys():
            df_patient_check_id = df_patient_check_group.get_group((doctor_logid,patient_logid))

            for index,row in df_patient_check_id.iterrows():
                creat_time = pd.to_datetime(row['time_text']).timestamp()
                if creat_time < starttime or creat_time > endtime:
                    continue

                click_item_parts = []

                # 如果变量不为 None，则添加到列表中
                if str(row['exam_type']) != 'nan':
                    click_item_parts.append(row['exam_type'])
                if str(row['imaging_type']) != 'nan':
                    click_item_parts.append(row['imaging_type'])
                if str(row['pathogen_type']) != 'nan':
                    click_item_parts.append(row['pathogen_type'])
                if str(row['culture_smear_type']) != 'nan':
                    click_item_parts.append(row['culture_smear_type'])

                # 将列表中的元素连接起来，并在每个非空元素之间添加下划线
                click_item = '下一步检查_' + '_'.join(click_item_parts)
                row_data = []
                row_data.append(interraction_id)
                row_data.append(click_item)
                row_data.append(creat_time)
                df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Clinician_Click_Behavior.csv', encoding='gbk', index=False)

