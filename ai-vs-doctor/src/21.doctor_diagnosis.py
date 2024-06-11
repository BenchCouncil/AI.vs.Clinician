import pandas as pd
from src.constant import *
import ast


columns = ['DOCTOR_ID', 'PATIENT_CASE_ID', 'DIAGNOSED_ORDER', 'WITH_MODEL', 'MODEL_INFER_ID', 'MODEL_VISIBILITY', 'STARTTIME','ENDTIME',
           'INTERACTION_ID', 'PRELIM_DIAGNOSIS','PRELIM_TREATMENT', 'PRELIM_TIMESTAMP',
           'FINAL_DIAGNOSIS','FINAL_TREATMENT', 'FINAL_TIMESTAMP','ACTION_TYPE',  'CLINICAL_TIME']

#doctor_id就是日志中医生的id，没变，patien_case_id可以根据sample的时间段获取


#获取开始诊断 结束诊断的时间
def get_start_endtime(df_sys_log_id,df_doctor_diag_id):
    df_diag_last = df_doctor_diag_id.sort_values(by='time',ascending=False)
    #最终诊断的结束时间应该是包括修改了诊断的时间 考虑到有的医生可能第二天又修改了诊断结果，但这时候的诊断时间并不是连续性的,只求当天的
    endtime = pd.to_datetime(df_diag_last.iloc[0]['time_text'])

    df_sys_log_id.loc[:, 'create_time'] = pd.to_datetime(df_sys_log_id['create_time'],format='%Y/%m/%d %H:%M')
    df_sys_log_id = df_sys_log_id[df_sys_log_id['create_time'] > endtime-pd.Timedelta(hours=1)]
    df_sys_log_id = df_sys_log_id[df_sys_log_id['module'] != '注销登录']

    df_sys_last = df_sys_log_id.sort_values(by='create_time', ascending=True)
    starttime = df_sys_last.iloc[0]['create_time']

    #按照分钟算
    start_stamp = int(starttime.timestamp())
    end_stamp = int(endtime.timestamp())

    time_diff = (end_stamp - start_stamp)/60

    # 最终诊断时间的均值是3.8分钟，筛选条件如下：和诊断时间模型的筛选条件一样
    max_value = 15.2  # 均值放大4倍
    min_value = 0.76  # 均值的20%
    if time_diff > max_value or time_diff < min_value:
        time_quality = False
    else:
        time_quality = True

    return time_quality,start_stamp,end_stamp

def get_diag_time(df_sys_log_id,df_doctor_diag_id):
    df_doctor_diag_id = df_doctor_diag_id[df_doctor_diag_id['operation'] != '点击下一个患者时自动保存信息']
    df_diag_last = df_doctor_diag_id.sort_values(by='time',ascending=False)
    endtime = pd.to_datetime(df_diag_last.iloc[0]['time_text'])
    df_sys_log_id.loc[:, 'create_time'] = pd.to_datetime(df_sys_log_id['create_time'])
    df_sys_log_id = df_sys_log_id[df_sys_log_id['create_time'] > endtime-pd.Timedelta(hours=1)]
    df_sys_log_id = df_sys_log_id[df_sys_log_id['module'] != '注销登录']

    df_diag_last = df_sys_log_id.sort_values(by='create_time', ascending=True)
    starttime = df_diag_last.iloc[0]['create_time']

    #按照分钟算
    time_diff = (int(endtime.timestamp()) - int(starttime.timestamp()))/60
    # if time_diff < 1 or time_diff > 20:
    #     print(df_sys_log_id[['module','create_time']])
    #     print(df_doctor_diag_id['time_text'])
    #     print(f'！！诊断时间可能异常： {time_diff}分钟')
    return round(time_diff,2)

#②获取医生诊断的时间
def get_first_diag_time(df_sys_log_id,df_doctor_diag_id):
    df_diag_id = df_doctor_diag_id[df_doctor_diag_id['operation'] != '点击下一个患者时自动保存信息']
    df_diag_id = df_diag_id[df_diag_id['final_diag'].isnull()]

    df_diag_last = df_diag_id.sort_values(by='time',ascending=False)

    if len(df_diag_last) == 0:
        openerate_set = df_doctor_diag_id['operation']
        if '修改了初步诊断' in openerate_set:
            df = df_doctor_diag_id[df_doctor_diag_id['operation'] == '修改了初步诊断']
        else:
            df = df_doctor_diag_id[df_doctor_diag_id['primary_diag'].notnull()]
        endtime = pd.to_datetime(df.iloc[0]['time_text'])
    else:
        endtime = pd.to_datetime(df_diag_last.iloc[0]['time_text'])
    df_sys_log_id.loc[:, 'create_time'] = pd.to_datetime(df_sys_log_id['create_time'])
    df_sys_log_id = df_sys_log_id[df_sys_log_id['create_time'] > endtime-pd.Timedelta(hours=1)]
    df_sys_log_id = df_sys_log_id[df_sys_log_id['module'] != '注销登录']

    df_diag_last = df_sys_log_id.sort_values(by='create_time', ascending=True)

    starttime = df_diag_last.iloc[0]['create_time']

    #按照分钟算
    time_diff = (int(endtime.timestamp()) - int(starttime.timestamp()))/60
    # if time_diff < 1 or time_diff > 20:
    #     print(df_sys_log_id[['module','create_time']])
    #     print(df_doctor_diag_id['time_text'])
    #     print(f'！！诊断时间可能异常： {time_diff}分钟')
    return round(time_diff,2)


def get_model(patient_case_id,df_sample_id,df_model_infer,df_model_property):
    model_result = df_sample_id.iloc[0]['AI模型预测结果']
    if str(model_result) == 'nan':
        with_model = False
    else:
        with_model = True

    model_infer_id = None
    model_visibility = None

    if with_model == True:
        df_model_property_id = None
        if 'RandomModel' in model_result:
            df_model_property_id = df_model_property[df_model_property['MODEL_NAME'] == 'RandomModel']
        elif 'TREWScore' in model_result:
            df_model_property_id = df_model_property[df_model_property['MODEL_NAME'] == 'CoxPHM']
        elif 'LSTM_AUC75' in model_result:
            df_model_property_id = df_model_property[df_model_property['MODEL_NAME'] == 'Low_LSTM']
        elif 'LSTM_AUC85' in model_result:
            df_model_property_id = df_model_property[df_model_property['MODEL_NAME'] == 'Medium_LSTM']
        elif 'LSTM_AUC95' in model_result:
            df_model_property_id = df_model_property[df_model_property['MODEL_NAME'] == 'High_LSTM']
        model_id = df_model_property_id.iloc[0]['MODEL_ID']
        if 'TREWScore' in model_result:
            model_result = model_result.replace(":", ":'", 1).replace(",", "',", 1)
            model_result = model_result.replace("No", "'No'")
            model_result = model_result.replace("Yes", "'Yes'")
        model_result = ast.literal_eval(model_result)
        for key in list(model_result.keys()):
            if 'IsVisible' in key:
                model_visibility = model_result.get(key)
        # print(model_id)
        # print(patient_case_id)
        df_model_infer_id = df_model_infer[(df_model_infer['MODEL_ID'] == model_id)&(df_model_infer['PATIENT_CASE_ID'] == patient_case_id)]
        model_infer_id = df_model_infer_id.iloc[0]['MODEL_INFER_ID']

    return with_model,model_infer_id,model_visibility

def diag_sequence(df_doctor_diag, doc_id,pat_id):

    df_doctor_diag_perdoc = df_doctor_diag[df_doctor_diag['doctor_id'] == doc_id]
    df_doctor_diag_perdoc = df_doctor_diag_perdoc.drop_duplicates(subset=['doctor_id', 'patient_id'])
    df_doctor_diag_perdoc = df_doctor_diag_perdoc.sort_values(by='id', ascending=True)
    df_doctor_diag_perdoc = df_doctor_diag_perdoc.assign(diag_sequence=range(1, len(df_doctor_diag_perdoc) + 1))
    # print(df_doctor_diag_perdoc)

    df_doctor_diag_pat = df_doctor_diag_perdoc[df_doctor_diag_perdoc['patient_id'] == pat_id]
    diag_sequence = df_doctor_diag_pat.iloc[0]['diag_sequence']
    return diag_sequence

def convert_diag(text):
    label_dict = {
        '无脓毒症': 0,
        '低度疑似脓毒症': 1,
        '高度疑似脓毒症': 1,
        '一般脓毒症': 1,
        '严重脓毒症': 1
    }
    matched_value = None

    # 遍历字典的键
    for key in label_dict:
        if key in text:
            matched_value = label_dict[key]
            break
    return matched_value

def get_first_final_diag(df_doctor_diag_id):
    df_doctor_diag_id = df_doctor_diag_id.sort_values(by='time', ascending=False)
    first_diag = df_doctor_diag_id.iloc[0]['primary_diag']
    final_diag = df_doctor_diag_id.iloc[0]['final_diag']
    return str(first_diag).strip(),str(final_diag).strip()


def data_validate(doctor_id,df_diag_id,df_doctor_info):
    #数据验证规则
    first_diag,final_diag = get_first_final_diag(df_diag_id)
    #1.删除初步诊断或最终诊断为null的情况
    if final_diag == 'nan' or first_diag == 'nan':
        return False
    #2.筛选初步和最终诊断至少有一个 sepsis相关
    first_diag_sepsis = convert_diag(first_diag)
    final_diag_sepsis = convert_diag(final_diag)
    if first_diag_sepsis is None and final_diag_sepsis is None:
        return False
    #3.最终诊断时间大于初步诊断时间
    final_diag_time = get_diag_time(df_sys_log_id, df_diag_id)
    first_diag_time = get_first_diag_time(df_sys_log_id, df_diag_id)
    if final_diag_time < first_diag_time:
        return False
    #4.医生信息不为空
    doctor_id_set = set(df_doctor_info['DOCTOR_ID'])
    if doctor_id not in doctor_id_set:
        return False

    # #③弃用：诊断时间合理:一起考虑的时候数据有点少
    # if time_quality is False:
    #     return False
    return True



if __name__ == '__main__':
    df = pd.DataFrame(columns=columns, dtype='object')

    df_patient_case = pd.read_csv(root + 'datasets\\Patient_Case_Info.csv',encoding='gbk')
    df_model_infer = pd.read_csv(root + 'datasets\\Model_Cohort_Infer.csv',encoding='gbk')
    df_model_property = pd.read_csv(root + 'datasets\\Model_Property.csv',encoding='gbk')
    df_doctor_info = pd.read_csv(root + 'datasets\\Doctor_Info.csv',encoding='gbk')

    df_sample = pd.read_csv(root + 'data\\样例数据_20231120.csv',encoding='gbk')

    #patientid是10000多的都是测试数据
    df_doctor_diag = pd.read_csv(root + 'data\\doctor\\doctor_diag.csv')
    df_doctor_diag = df_doctor_diag[(df_doctor_diag['patient_id'] <= 6000) | (df_doctor_diag['patient_id'] > 20000)]
    df_doctor_diag_group = df_doctor_diag.groupby(['doctor_id', 'patient_id'])

    df_syslog = pd.read_csv(root + 'data\\doctor\\sys_log.csv')
    df_syslog = df_syslog[(df_syslog['accountname'] <= 6000) | (df_syslog['patient_id'] > 20000)]

    data_num = 0

    interraction_id = None
    for (doctor_id,patient_id),df_diag_id in df_doctor_diag_group:
        interraction_id = f'{doctor_id}_{patient_id}'
        if patient_id > 20000:
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            patient_id = patient_id + 20000
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]

        df_patient_case_id = df_patient_case[df_patient_case['CURRENT_TIME'] == df_sample_id.iloc[0]['START_ENDTIME']]
        patient_case_id = df_patient_case_id.iloc[0]['PATIENT_CASE_ID']

        order = diag_sequence(df_doctor_diag, doctor_id, patient_id)
        with_model,model_predict_id,model_visibility = get_model(patient_case_id,df_sample_id,df_model_infer,df_model_property)

        df_diag_id = df_diag_id.sort_values(by='time')

        df_first_diag_id = df_diag_id[df_diag_id['final_diag'].isnull()]
        df_final_diag_id = df_diag_id[~df_diag_id['final_diag'].isnull()]
        if len(df_first_diag_id)+ len(df_final_diag_id) != len(df_diag_id):
            print('check first diag and final diag ')

        df_sys_log_id = df_syslog[(df_syslog['accountname'] == doctor_id) & (df_syslog['patient_id'] == patient_id)]
        df_final_diag_id = df_final_diag_id[df_final_diag_id['operation'] != '点击下一个患者时自动保存信息']

        if len(df_final_diag_id) != 0:
            time_quality,starttime,endtime = get_start_endtime(df_sys_log_id,df_final_diag_id)
        else:
            time_quality,starttime,endtime = get_start_endtime(df_sys_log_id,df_first_diag_id)

        #数据验证规则：诊断质量不高的直接跳过，不记录到数据库
        val_result = data_validate(doctor_id,df_diag_id,df_doctor_info)
        if val_result is True:
            data_num = data_num+1

        if val_result:
            diag_modify_count = 0

            for index,row in df_first_diag_id.iterrows():
                row_data = []
                row_data.append(doctor_id)
                row_data.append(patient_case_id)
                row_data.append(order)
                row_data.append(with_model)
                row_data.append(model_predict_id)
                row_data.append(model_visibility)
                row_data.append(starttime)
                row_data.append(endtime)

                row_data.append(interraction_id)
                row_data.append(row['primary_diag'])
                row_data.append(row['primary_med'])
                row_data.append(pd.to_datetime(row['time_text']).timestamp())
                row_data.append(None)
                row_data.append(None)
                row_data.append(None)
                if diag_modify_count == 0:
                    row_data.append('add primary diagnosis')
                else:
                    row_data.append('modify primary diagnosis')
                row_data.append(None)
                df.loc[len(df)] = row_data
                diag_modify_count += 1

            diag_modify_count = 0
            for index,row in df_final_diag_id.iterrows():
                row_data = []
                row_data.append(doctor_id)
                row_data.append(patient_case_id)
                row_data.append(order)
                row_data.append(with_model)
                row_data.append(model_predict_id)
                row_data.append(model_visibility)
                row_data.append(starttime)
                row_data.append(endtime)

                row_data.append(interraction_id)
                if row['operation'] == '修改了初步诊断':
                    row_data.append(row['primary_diag'])
                    row_data.append(row['primary_med'])
                    row_data.append(pd.to_datetime(row['time_text']).timestamp())
                    row_data.append(None)
                    row_data.append(None)
                    row_data.append(None)
                    row_data.append('modify primary diagnosis')
                else:
                    row_data.append(None)
                    row_data.append(None)
                    row_data.append(None)
                    row_data.append(row['final_diag'])
                    row_data.append(row['final_med'])
                    row_data.append(pd.to_datetime(row['time_text']).timestamp()) #can not use row['time_text']
                    if diag_modify_count == 0: #log中修改了最终诊断  其实是添加最终诊断
                        row_data.append('add final diagnosis')
                    else:
                        row_data.append('modify final diagnosis')
                row_data.append(row['cost_time'])
                df.loc[len(df)] = row_data
                diag_modify_count += 1
    print(f'data num is {data_num}')

    df.to_csv(root + '\\datasets\\Doctor_Diagnosis_Treatment.csv', encoding='gbk', index=False)
