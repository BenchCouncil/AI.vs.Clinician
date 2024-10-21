# -*- coding: utf-8 -*-
from src.log_abstract import *
from src.sample_read import *
pd.options.mode.chained_assignment = None  # default='warn'

doctor_logid = '系统日志中医生ID'
doctor_unit_dict = {
    # 二甲医院
    "1": 1,
    # 三甲医院
    "2": 2,
    # 医学院
    "3": 3,
}
def get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort, is_display):
    if model_sort == '':
        return doctor_patient_set

    if model_sort == '无模型':
        df_model = df_sample[~df_sample['AI模型预测结果'].notna()]
    else:
        df_sample = df_sample[df_sample['AI模型预测结果'].notna()]
        df_model = df_sample[df_sample['AI模型预测结果'].str.contains(model_sort)]
        if is_display != '':
            df_model = df_model[df_model['AI模型预测结果'].str.contains(is_display)]

    # ① 分类：随机模型
    patient_id_random = set(df_model['UNIQUE_ID'])
    # ② 确定随机模型 下 患者的所有医生

    doctor_patient_random = set()
    for doctor_patient in doctor_patient_set:
        patient_id = doctor_patient[1]
        # 处理一个患者可能被多个医生诊断的情况
        if dataset == 12000:
            if patient_id >= 20000:
                if patient_id - 20000 in patient_id_random:
                    doctor_patient_random.add(doctor_patient)
        if patient_id <= 6000:
            if patient_id in patient_id_random:
                doctor_patient_random.add(doctor_patient)
    return doctor_patient_random


def sample_num_subjectid_num(dataset, df_sample, doctor_patient_set, model_sort, is_display):
    doctor_patient_temp = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort, is_display)

    sample_num = len(doctor_patient_temp)
    subject_id_set = set()
    for doctor_patient in doctor_patient_temp:
        doctor_id = doctor_patient[0]
        patient_id = doctor_patient[1]
        if patient_id > 20000:
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            patient_id = patient_id + 20000
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        subject_id = df_sample_id.iloc[0]['SUBJECT_ID']
        subject_id_set.add(subject_id)
    subject_id_num = len(subject_id_set)

    print(subject_id_num)#患者数
    # print(sample_num)#诊断样例数据




# 不同模型诊断准确率  不分医生一起算准确率
def model_exper1(diag_flag, dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort, is_display)

    diag_acc_list= []
    for doctor_patient in doctor_patient_random:
        doctor_id = doctor_patient[0]
        patient_id = doctor_patient[1]
        if patient_id > 20000:
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            patient_id = patient_id + 20000
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        df_doctor_diag_id = df_doctor_diag[
            (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]

        # 改变这个 为 最终诊断
        if diag_flag == 'first':
            diag = get_first_diag(df_doctor_diag_id)
        else:
            diag = get_final_diag(df_doctor_diag_id)
        if len(df_sample_id) != 0:
            diag_acc = get_diag_acc(df_sample_id, diag)
            if diag_acc is not None:
                diag_acc_list.append(diag_acc)
    if len(diag_acc_list) == 0:
        avg_diag_acc = '没有诊断样例'
        auc_ci= None
    else:
        avg_diag_acc = round((sum(diag_acc_list) / len(diag_acc_list))*100, 2) #把这里的*100也可以去掉
        auc_ci = cal_ci(diag_acc_list)

    # print(f'数据集规模为{dataset}，在{diag_flag}诊断模型中，{model_sort} 显示情况{is_display} 的平均诊断准确率为 {diag_acc}')
    print(avg_diag_acc)
    print(auc_ci)
    # print(len(diag_acc_list))
    # ③ 对每个医生诊断每个患者的算准确率
    # ④ 患者的csv中患者的一行数据（主要是获取患病和时间段 -3h 0h 3h）
    # ⑤ 医生的诊断结果
    # ⑥ 诊断准确率 输入：sample的一行 和 医生的诊断结果


# 不同模型诊断准确率  分每个医生算准确率然后平均
def model_exper1_perdocavg(diag_flag, dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_doctor_diag_part = df_doctor_diag[
        (df_doctor_diag['doctor_id'].isin(doctor_id_set_part)) & (
            df_doctor_diag['patient_id'].isin(patient_id_set_part))]
    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id'])
    doctors_acc_list = []
    for index, row_doctor_diag in df_doctor_diag_group:
        doctor_acc = get_diag_acc_perdoctor(df_sample, row_doctor_diag, diag_flag)
        if doctor_acc is not None:
            doctors_acc_list.append(doctor_acc)
    if len(doctors_acc_list) == 0:
        acc_avge = '没有诊断样例'
    else:
        acc_avge = round((sum(doctors_acc_list) / len(doctors_acc_list)) * 100, 2)
    # print(f'数据集规模为{dataset}，在{diag_flag}诊断模型中，{model_sort} 显示情况{is_display} 的医生诊断结果中平均ACC {acc} %')
    print(acc_avge)


# 不同模型诊断时间
def model_exper2(diag_flag, dataset, df_sample, doctor_patient_set, df_doctor_diag, df_syslog, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    value_list = []
    for doctor_patient in doctor_patient_random:
        doctor_id = doctor_patient[0]
        patient_id = doctor_patient[1]
        if patient_id > 20000:
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            patient_id = patient_id + 20000
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        df_doctor_diag_id = df_doctor_diag[
            (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]
        df_sys_log_id = df_syslog[(df_syslog['accountname'] == doctor_id) & (df_syslog['patient_id'] == patient_id)]

        if diag_flag == 'first':
            diagtime = get_first_diag_time(df_sys_log_id, df_doctor_diag_id)
        else:
            diagtime = get_final_diag_time(df_sys_log_id, df_doctor_diag_id)
        if diagtime < 30:  # 如果诊断时间大于30的话，可能存在异常，暂时不纳入统计分析
            value_list.append(diagtime)
    if len(value_list) == 0:
        value_avg = '没有诊断样例'
        valie_ci= None
    else:
        value_avg = round((sum(value_list) / len(value_list)), 2)
        valie_ci = cal_ci(value_list)
    # print(f'数据集规模为{dataset}，在{diag_flag}诊断模型中，{model_sort} 显示情况{is_display} 的平均诊断时间为 {value_avg}分钟')
    print(f'{value_avg}')
    print(valie_ci)
    return value_avg



# 不同模型诊断修改次数
def model_exper3(diag_flag, dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    value_list = []
    for doctor_patient in doctor_patient_random:
        doctor_id = doctor_patient[0]
        patient_id = doctor_patient[1]

        df_doctor_diag_id = df_doctor_diag[
            (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]

        if diag_flag == 'first':
            edit_count = first_diag_edit_counts(df_doctor_diag_id)
        else:
            edit_count = final_diag_edit_counts(df_doctor_diag_id)
        value_list.append(edit_count)
    if len(value_list) == 0:
        value_avg = '没有诊断样例'
        value_ci = None
    else:
        value_avg = round((sum(value_list) / len(value_list)), 2)
        value_ci = cal_ci(value_list)
    # print(f'数据集规模为{dataset}，在{diag_flag}诊断模型中，{model_sort} 显示情况{is_display} 的平均修改次数为 {value_avg}次')
    print(value_avg)
    print(value_ci)
    return value_avg


# 不同模型医生查看行为的对比
def model_exper4(dataset, df_sample, doctor_patient_set, df_doctor_diag, df_patient_check, model_sort, is_display,
                 fold_name):
    if model_sort == 'all':
        doctor_patient_random = doctor_patient_set
    else:
        doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                                 is_display)
    doctor_set = set()
    model_patient_set = set()
    for item in doctor_patient_random:
        doctor_set.add(item[0])
        model_patient_set.add(item[1])

    for doctor_id in doctor_set:
        df_doctor = df_doctor_diag[(df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['operation'] == '新增诊断')]
        df_doctor = df_doctor.sort_values(by='time', ascending=True)
        patient_id_set = set(df_doctor['patient_id'])
        patient_id_set = patient_id_set & model_patient_set

        df_patient_check_id = df_patient_check[df_patient_check['doctor_id'] == doctor_id]
        doctor_view_check(doctor_id, patient_id_set, df_patient_check_id, fold_name)


# 不同模型的抗生素使用正确率
def model_exper5_antacc(diag_flag, dataset, df_sample, doctor_patient_set, df_doctor_diag, all_drugs_list, model_sort,
                 is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)

    antuse_acc_list = []
    for doctor_patient in doctor_patient_random:
        doctor_id = doctor_patient[0]
        patient_id = doctor_patient[1]
        if patient_id > 20000:
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            patient_id = patient_id + 20000
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        df_doctor_diag_id = df_doctor_diag[
            (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]
        df_doctor_diag_id = df_doctor_diag_id.sort_values(by='time', ascending=False)
        first_med_list = df_doctor_diag_id.iloc[0]['primary_med']
        final_med_list = df_doctor_diag_id.iloc[0]['final_med']

        if diag_flag == 'first':
            antacc = antuse_acc_v2(df_sample_id, all_drugs_list, first_med_list)
        else:
            antacc = antuse_acc_v2(df_sample_id, all_drugs_list, final_med_list)

        if antacc is not None:
            antuse_acc_list.append(antacc)

    if len(antuse_acc_list) == 0:
        avg_antuse_acc = '没有诊断样例'
        antuse_acc_ci = None
    else:
        avg_antuse_acc = round(sum(antuse_acc_list)/len(antuse_acc_list)*100, 2)
        antuse_acc_ci = cal_ci(antuse_acc_list)

    # print(f'数据集规模为{dataset}，在{diag_flag}诊断模型中，{model_sort} 显示情况{is_display} 的平均抗生素使用准确率为 {antac} %')
    print(avg_antuse_acc)
    print(antuse_acc_ci)
    return avg_antuse_acc


before_diag_hour_dict = {
    '3h': -3,
    '2h': -2,
    '1h': -1,
    '0h': 0,
    '-1h': 1,
    '-2h': 2,
    '-3h':3
}
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

#诊断出sepsis的减少时间
def model_exper_get_before_diag_hours(diag_flag,dataset,df_sample_timerange,doctor_patient_set,df_doctor_diag,model_sort,is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample_timerange, doctor_patient_set, model_sort,is_display)

    model_vis_patset = set()
    for item in doctor_patient_random:
        model_vis_patset.add(item[1])

    before_hour_list = []
    unique_id_set = set(df_sample_timerange['UNIQUE_ID'])
    unique_id_20000 = {item + 20000 for item in unique_id_set}
    unique_id_set.update(unique_id_20000)
    patient_set = {item[1] for item in doctor_patient_random}
    patient_inter_set = unique_id_set.intersection(patient_set)
    df_sample_group1_sepsis_after = df_sample_timerange[
        df_sample_timerange['UNIQUE_ID'].isin(patient_inter_set)]
    doctor_patient_set_after = [item for item in doctor_patient_random if item[1] in patient_inter_set]

    df_group_hadmid = df_sample_group1_sepsis_after.groupby('HADM_ID')
    for hadm_id, df_row in df_group_hadmid:
        diag_sepsis_timedetail_set = set()
        # hadm_id has many diag,with diff model
        uniqueid_list = set(df_row['UNIQUE_ID'])
        # doctor_patient_set_after_temp = [item for item in doctor_patient_set_after if item[1] in uniqueid_list] #this is one sample only one doctor diag
        doctor_patient_set_after_temp = [item for item in doctor_patient_set_after if
                                         (item[1] in uniqueid_list) or (item[1] - 20000 in uniqueid_list)]

        for doctor_patient in doctor_patient_set_after_temp:
            doctor_id = doctor_patient[0]
            patient_id = doctor_patient[1]
            if (patient_id - 20000 not in model_vis_patset) and (
                    patient_id not in model_vis_patset):  # current model patid flit
                continue

            # check diag has sepsis
            df_doctor_diag_id = df_doctor_diag[
                (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]
            if diag_flag == 'first':
                diag = get_first_diag(df_doctor_diag_id)
            else:
                diag = get_final_diag(df_doctor_diag_id)
            diag_int = convert_diag(diag)

            if diag_int is not None:
                if diag_int == 1:
                    if patient_id > 20000:
                        patient_id = patient_id - 20000
                    df_detail_id = df_sample_timerange[(df_sample_timerange['UNIQUE_ID'] == patient_id)]
                    time_detail = df_detail_id.iloc[0]['time_range_detail']
                    diag_sepsis_timedetail_set.add(time_detail)

        # find time_range_detail list by uniqueid for dias is sepsis
        if len(diag_sepsis_timedetail_set) > 0:
            for key in before_diag_hour_dict.keys():  # 取出最开始诊断sepsis的时间
                if key in diag_sepsis_timedetail_set:
                    before_hour_list.append(before_diag_hour_dict.get(key))
                    break

    if len(before_hour_list) == 0:
        avg_before_hour = '没有样本 or 医生没有诊断出来是脓毒症患者'
    else:
        avg_before_hour = round(sum(before_hour_list) / len(before_hour_list), 4)

    print(avg_before_hour)


def model_exper_get_before_antuse_hours(diag_flag,dataset,df_sample_timerange,doctor_patient_set,df_doctor_diag,ant_name_list,model_sort,is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample_timerange, doctor_patient_set, model_sort,is_display)

    model_vis_patset = set()
    for item in doctor_patient_random:
        model_vis_patset.add(item[1])

    before_hour_list = []
    unique_id_set = set(df_sample_timerange['UNIQUE_ID'])
    unique_id_20000 = {item + 20000 for item in unique_id_set}
    unique_id_set.update(unique_id_20000)
    patient_set = {item[1] for item in doctor_patient_random}
    patient_inter_set = unique_id_set.intersection(patient_set)
    df_sample_ill_after = df_sample_timerange[df_sample_timerange['UNIQUE_ID'].isin(patient_inter_set)]
    doctor_patient_set_after = [item for item in doctor_patient_random if item[1] in patient_inter_set]

    df_group_hadmid = df_sample_ill_after.groupby('HADM_ID')
    for hadm_id, df_row in df_group_hadmid:
        sepsis_use_ant_timedetail_set = set()
        # hadm_id has many diag,with diff model
        uniqueid_list = set(df_row['UNIQUE_ID'])
        doctor_patient_set_after_temp = [item for item in doctor_patient_set_after if
                                         (item[1] in uniqueid_list) or (item[1] - 20000 in uniqueid_list)]

        for doctor_patient in doctor_patient_set_after_temp:
            doctor_id = doctor_patient[0]
            patient_id = doctor_patient[1]
            if (patient_id - 20000 not in model_vis_patset) and (
                    patient_id not in model_vis_patset):  # current model patid flit
                continue

            # check sepsis patient's med has ant
            df_doctor_diag_id = df_doctor_diag[
                (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]

            df_doctor_diag_id = df_doctor_diag_id.sort_values(by='time', ascending=False)
            first_med_list = df_doctor_diag_id.iloc[0]['primary_med']
            final_med_list = df_doctor_diag_id.iloc[0]['final_med']
            if diag_flag == 'first':
                if str(first_med_list) == 'nan':
                    continue
                contains_any_ant = any(element in first_med_list for element in ant_name_list)
            else:
                if str(final_med_list) == 'nan':
                    continue
                contains_any_ant = any(element in final_med_list for element in ant_name_list)

            if contains_any_ant is True:  # take down time_detail when doctor med contain ant
                if patient_id > 20000:
                    patient_id = patient_id - 20000
                df_detail_id = df_sample_ill_after[(df_sample_ill_after['UNIQUE_ID'] == patient_id)]
                time_detail = df_detail_id.iloc[0]['time_range_detail']
                sepsis_use_ant_timedetail_set.add(time_detail)

        # find time_range_detail list by uniqueid for dias is sepsis

        if len(sepsis_use_ant_timedetail_set) > 0:
            for key in before_diag_hour_dict.keys():  # 取出最开始诊断sepsis的时间
                if key in sepsis_use_ant_timedetail_set:
                    before_hour_list.append(before_diag_hour_dict.get(key))
                    break

    if len(before_hour_list) == 0:
        avg_before_hour = '没有样本 or 医生没有给脓毒症患者使用抗生素'
    else:
        avg_before_hour = round(sum(before_hour_list) / len(before_hour_list), 4)

    print(avg_before_hour)





# 不同模型的医生诊断中补液剂量的差异
def model_exper6(diag_flag, dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    value_list = []
    for doctor_patient in doctor_patient_random:
        doctor_id = doctor_patient[0]
        patient_id = doctor_patient[1]

        df_doctor_diag_id = df_doctor_diag[
            (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]
        if diag_flag == 'first':
            fluid_count = first_diag_fluid(df_doctor_diag_id)
        else:
            fluid_count = final_diag_fluid(df_doctor_diag_id)
        if fluid_count is not None:
            value_list.append(fluid_count)
    if len(value_list) ==0:
        value_avg = '没有诊断样例'
    else:
        value_avg = round((sum(value_list) / len(value_list)), 2)
    # print(f'数据集规模为{dataset}，在{diag_flag}诊断模型中，{model_sort} 显示情况{is_display} 的医生诊断结果中平均补液 {value_avg} ml')
    print(value_avg)


# 不同模型的医生诊断auc的差异
def model_exper7_auc(diag_flag, dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    # doctor_patient_random = doctor_patient_set #先尝试一下，不按照模型划分呢

    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_doctor_diag_part = df_doctor_diag[
        (df_doctor_diag['doctor_id'].isin(doctor_id_set_part)) & (
            df_doctor_diag['patient_id'].isin(patient_id_set_part))]

    avg_auc,auc_ci = diag_avg_auc(df_sample, df_doctor_diag_part, diag_flag)
    # print(f'数据集规模为{dataset}，在{diag_flag}诊断模型中，{model_sort} 显示情况{is_display} 的医生诊断结果中平均AUC {avg_auc} %')
    print(avg_auc)
    print(auc_ci)
    return avg_auc


# 不同模型的医生诊断时间修改率的差异
def model_exper8(diag_flag, dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_doctor_diag_part = df_doctor_diag[
        (df_doctor_diag['doctor_id'].isin(doctor_id_set_part)) & (
            df_doctor_diag['patient_id'].isin(patient_id_set_part))]

    modify_rate = diag_modify_rate(df_doctor_diag_part, diag_flag)

    # print(f'数据集规模为{dataset}，在{diag_flag}诊断模型中，{model_sort} 显示情况{is_display} 的医生诊断结果修改率 {modify_rate} %')
    print(modify_rate)


# 不同模型的医生诊断时间修改的差异
def model_exper9(dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_doctor_diag_part = df_doctor_diag[
        (df_doctor_diag['doctor_id'].isin(doctor_id_set_part)) & (
            df_doctor_diag['patient_id'].isin(patient_id_set_part))]

    change_list = diag_change(df_doctor_diag_part)
    # print(f'数据集规模为{dataset}，{model_sort} 显示情况{is_display} 的医生诊断结果修改变化统计 {change_list} ')
    # print(change_list)
    print(change_list[0]) #总的修改次数

def model_exper9_acc(dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_doctor_diag_part = df_doctor_diag[
        (df_doctor_diag['doctor_id'].isin(doctor_id_set_part)) & (
            df_doctor_diag['patient_id'].isin(patient_id_set_part))]

    diag_change_avge_acc,diag_change_true_num = diag_change_acc(df_sample,df_doctor_diag_part)
    # print(f'数据集规模为{dataset}，{model_sort} 显示情况{is_display} 的医生诊断结果变化的平均准确率 {diag_change_avge_acc} ')
    # print(diag_change_true_num) #诊断结果变化的正确的数量
    print(diag_change_avge_acc) #诊断结果变化的准确率


# 不同模型的初步到最终抗生素的使用变化
def model_exper10(all_antname_list, dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_doctor_diag_part = df_doctor_diag[
        (df_doctor_diag['doctor_id'].isin(doctor_id_set_part)) & (
            df_doctor_diag['patient_id'].isin(patient_id_set_part))]

    change_list = antibio_change(df_doctor_diag_part, all_antname_list)
    # print(f'数据集规模为{dataset}，{model_sort} 显示情况{is_display} 的初步到最终抗生素的使用变化 {change_list} ')
    # print(change_list)
    print(sum(change_list)) #总的抗生素使用变化 次数

# 不同模型的初步到最终抗生素的使用变化的正确率
def model_exper10_acc(all_antname_list, dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_doctor_diag_part = df_doctor_diag[
        (df_doctor_diag['doctor_id'].isin(doctor_id_set_part)) & (
            df_doctor_diag['patient_id'].isin(patient_id_set_part))]

    antibio_change_avge_acc,antibio_change_true_num = antibio_change_acc(df_sample,df_doctor_diag_part, all_antname_list)
    # print(f'数据集规模为{dataset}，{model_sort} 显示情况{is_display} 的初步到最终抗生素的使用变化 {change_list} ')
    # print(antibio_change_true_num) #初步到最终抗生素的使用变化的正确的数量
    print(antibio_change_avge_acc)  # 初步到最终抗生素的使用变化的准确率


# 不同模型的初步到最终的补液变化
def model_exper11(dataset, df_sample, doctor_patient_set, df_doctor_diag, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_doctor_diag_part = df_doctor_diag[
        (df_doctor_diag['doctor_id'].isin(doctor_id_set_part)) & (
            df_doctor_diag['patient_id'].isin(patient_id_set_part))]

    aver_up, aver_down = fluid_change(df_doctor_diag_part)
    # print(f'数据集规模为{dataset}，{model_sort} 显示情况{is_display} 的初步到最终补液变化 向上平均变化 {aver_up} ml,向下平均变化 {aver_down} ml ')
    if aver_up == '没有诊断样例':
        print('没有诊断样例')
    else:
        print(f'增加{aver_up}ml,减少{aver_down}ml ')


# 患者被查看的检查数量（历史检查、下一步检查）
def model_exper12_aver_hisnum(dataset, df_sample, doctor_patient_set, df_patient_check, df_syslog, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_patient_check_part = df_patient_check[(df_patient_check['doctor_id'].isin(doctor_id_set_part)) & (
        df_patient_check['patient_id'].isin(patient_id_set_part))]
    df_syslog_part = df_syslog[
        (df_syslog['accountname'].isin(doctor_id_set_part)) & (df_syslog['patient_id'].isin(patient_id_set_part))]

    aver_hisnum, aver_nextnum,value_ci_his,value_ci_next = view_check_num(doctor_patient_random, df_syslog_part, df_patient_check_part)
    # print(f'数据集规模为{dataset}，{model_sort} 显示情况{is_display} 的患者被查看检查 历史检查平均 {aver_hisnum} 项,下一步检查平均 {aver_nextnum} 项 ')
    print(aver_hisnum)
    print(value_ci_his)
    return aver_hisnum



# 患者被查看的检查数量（历史检查、下一步检查）
def model_exper12_aver_nextnum(dataset, df_sample, doctor_patient_set, df_patient_check, df_syslog, model_sort, is_display):
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, model_sort,
                                                             is_display)
    doctor_id_set_part = set()
    patient_id_set_part = set()
    for doctor_patient in doctor_patient_random:
        doctor_id_set_part.add(doctor_patient[0])
        patient_id_set_part.add(doctor_patient[1])

    df_patient_check_part = df_patient_check[(df_patient_check['doctor_id'].isin(doctor_id_set_part)) & (
        df_patient_check['patient_id'].isin(patient_id_set_part))]
    df_syslog_part = df_syslog[
        (df_syslog['accountname'].isin(doctor_id_set_part)) & (df_syslog['patient_id'].isin(patient_id_set_part))]

    aver_hisnum, aver_nextnum,value_ci_his,value_ci_next = view_check_num(doctor_patient_random, df_syslog_part, df_patient_check_part)
    # print(f'数据集规模为{dataset}，{model_sort} 显示情况{is_display} 的患者被查看检查 历史检查平均 {aver_hisnum} 项,下一步检查平均 {aver_nextnum} 项 ')
    print(aver_nextnum)
    print(value_ci_next)


# 患者分为脓毒症患者、非脓毒症患者
def patient_exper1(df_sample, doctor_patient_set):
    sepsis_doctor_patient_set = set()
    notsepsis_doctor_patient_set = set()

    df_sample_sepsis = df_sample[df_sample['ILL_TIME'].notna()]
    df_sample_notsepsis = df_sample[~df_sample['ILL_TIME'].notna()]

    sepsis_id_set = set(df_sample_sepsis['UNIQUE_ID'])
    notsepsis_id_set = set(df_sample_notsepsis['UNIQUE_ID'])

    for (doctor_id, patient_id) in doctor_patient_set:
        if patient_id > 20000:
            patient_id_temp = patient_id - 20000
        else:
            patient_id_temp = patient_id
        if patient_id_temp in sepsis_id_set:
            sepsis_doctor_patient_set.add((doctor_id, patient_id))
        if patient_id_temp in notsepsis_id_set:
            notsepsis_doctor_patient_set.add((doctor_id, patient_id))
    return sepsis_doctor_patient_set, notsepsis_doctor_patient_set

def patient_exper2(df_sample, doctor_patient_set):
    sepsis_3h_docpatset = set()
    sepsis_0h_docpatset = set()
    sepsis_3hback_docpatset = set()
    notsepsis_docpatset = set()

    df_sample_sepsis = df_sample[df_sample['ILL_TIME'].notna()]
    df_sample_sepsis_3h = df_sample_sepsis[df_sample_sepsis['TIME_RANGE'] == '3h']
    df_sample_sepsis_0h = df_sample_sepsis[df_sample_sepsis['TIME_RANGE'] == '0h']
    df_sample_sepsis_3hback = df_sample_sepsis[df_sample_sepsis['TIME_RANGE'] == '-3h']
    df_sample_notsepsis = df_sample[~df_sample['ILL_TIME'].notna()]

    sepsis_3h_id_set = set(df_sample_sepsis_3h['UNIQUE_ID'])
    sepsis_0h_id_set = set(df_sample_sepsis_0h['UNIQUE_ID'])
    sepsis_3hback_id_set = set(df_sample_sepsis_3hback['UNIQUE_ID'])
    notsepsis_id_set = set(df_sample_notsepsis['UNIQUE_ID'])

    for (doctor_id, patient_id) in doctor_patient_set:
        if patient_id > 20000:
            patient_id_temp = patient_id - 20000
        else:
            patient_id_temp = patient_id
        if patient_id_temp in sepsis_3h_id_set:
            sepsis_3h_docpatset.add((doctor_id, patient_id))
        elif patient_id_temp in sepsis_0h_id_set:
            sepsis_0h_docpatset.add((doctor_id, patient_id))
        elif patient_id_temp in sepsis_3hback_id_set:
            sepsis_3hback_docpatset.add((doctor_id, patient_id))
        elif patient_id_temp in notsepsis_id_set:
            notsepsis_docpatset.add((doctor_id, patient_id))
    return sepsis_3h_docpatset,sepsis_0h_docpatset,sepsis_3hback_docpatset,notsepsis_docpatset


def count_doc(dp_set):
    doctor_set = set()
    for item in dp_set:
        doctor_set.add(item[0])
    return len(doctor_set)

class_position_dict = {
    '无':'None (During residency training)',
    '住院医师':'Junior (Resident physician)',
    '初级':'Junior (Resident physician)',
    '主治': 'Intermediate (Attending physician)',
    '主治医师': 'Intermediate (Attending physician)',
    '主治医生': 'Intermediate (Attending physician)',
    '副主任医师': 'Senior (Chief and Associate Chief Physician)',
    '副主任': 'Senior (Chief and Associate Chief Physician)',
    '副主任医生': 'Senior (Chief and Associate Chief Physician)',
    '主任医师': 'Senior (Chief and Associate Chief Physician)',

}

# 按照医生职称分类:  无  初级  中级  副高级  高级
def doctor_exper1_position(doctor_patient_set, df_doctor_info):
    df_level1 = df_doctor_info[df_doctor_info['职称'] == '无']
    df_level2 = df_doctor_info[(df_doctor_info['职称'] == '住院医师') | (df_doctor_info['职称'] == '初级')]
    df_level3 = df_doctor_info[
        (df_doctor_info['职称'] == '主治') | (df_doctor_info['职称'] == '主治医师') | (df_doctor_info['职称'] == '主治医生')]
    df_level4 = df_doctor_info[
        (df_doctor_info['职称'] == '副主任医师') | (df_doctor_info['职称'] == '副主任') |(df_doctor_info['职称'] == '副主任医生') | (df_doctor_info['职称'] == '主任医师')]
    doctor_patient_set_level1 = set()
    doctor_patient_set_level2 = set()
    doctor_patient_set_level3 = set()
    doctor_patient_set_level4 = set()

    for (doctor_id, patient_id) in doctor_patient_set:
        if doctor_id in set(df_level1[doctor_logid]):
            doctor_patient_set_level1.add((doctor_id, patient_id))
        if doctor_id in set(df_level2[doctor_logid]):
            doctor_patient_set_level2.add((doctor_id, patient_id))
        if doctor_id in set(df_level3[doctor_logid]):
            doctor_patient_set_level3.add((doctor_id, patient_id))
        if doctor_id in set(df_level4[doctor_logid]):
            doctor_patient_set_level4.add((doctor_id, patient_id))
    # print(f'总计{count_doc(doctor_patient_set_level1)}') #用于计算分类后的医生数量 是不是和日志中医生总数相同
    return doctor_patient_set_level1, doctor_patient_set_level2, doctor_patient_set_level3, doctor_patient_set_level4


# 按照医生从业年限: 0-5  5-10  10-15  15-20  >20
def doctor_exper2_year(doctor_patient_set, df_doctor_info):
    df_age5 = df_doctor_info[df_doctor_info['从业年限'] <= 5]
    df_age10 = df_doctor_info[(df_doctor_info['从业年限'] > 5) & (df_doctor_info['从业年限'] <= 10)]
    df_age15 = df_doctor_info[(df_doctor_info['从业年限'] > 10) & (df_doctor_info['从业年限'] <= 15)]
    df_age20 = df_doctor_info[(df_doctor_info['从业年限'] > 15) & (df_doctor_info['从业年限'] <= 20)]
    df_age25 = df_doctor_info[(df_doctor_info['从业年限'] > 20)]

    doctor_patient_set_age5 = set()
    doctor_patient_set_age10 = set()
    doctor_patient_set_age15 = set()
    doctor_patient_set_age20 = set()
    doctor_patient_set_age25 = set()

    for (doctor_id, patient_id) in doctor_patient_set:
        if doctor_id in set(df_age5[doctor_logid]):
            doctor_patient_set_age5.add((doctor_id, patient_id))
        if doctor_id in set(df_age10[doctor_logid]):
            doctor_patient_set_age10.add((doctor_id, patient_id))
        if doctor_id in set(df_age15[doctor_logid]):
            doctor_patient_set_age15.add((doctor_id, patient_id))
        if doctor_id in set(df_age20[doctor_logid]):
            doctor_patient_set_age20.add((doctor_id, patient_id))
        if doctor_id in set(df_age25[doctor_logid]):
            doctor_patient_set_age25.add((doctor_id, patient_id))

    return doctor_patient_set_age5, doctor_patient_set_age10, doctor_patient_set_age15, doctor_patient_set_age20, doctor_patient_set_age25


# 按照医生性别:男  女
def doctor_exper3_gender(doctor_patient_set, df_doctor_info):
    df_gender_0 = df_doctor_info[df_doctor_info['性别'] == '女']
    df_gender_1 = df_doctor_info[df_doctor_info['性别'] == '男']
    doctor_patient_set_gender0 = set()
    doctor_patient_set_gender1 = set()
    for (doctor_id, patient_id) in doctor_patient_set:
        if doctor_id in set(df_gender_0[doctor_logid]):
            doctor_patient_set_gender0.add((doctor_id, patient_id))
        if doctor_id in set(df_gender_1[doctor_logid]):
            doctor_patient_set_gender1.add((doctor_id, patient_id))
    return doctor_patient_set_gender0, doctor_patient_set_gender1

def modelsort_exper(dataset, df_sample, doctor_patient_set):
    doctor_patient_nomodel = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, '无模型','No')
    doctor_patient_random = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'RandomModel','No')
    doctor_patient_coxm_blinded = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'TREWScore','No')
    doctor_patient_coxm_non_blinded = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'TREWScore','Yes')
    doctor_patient_lstm75_blinded = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'LSTM_AUC75','No')
    doctor_patient_lstm75_non_blinded = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'LSTM_AUC75','Yes')
    doctor_patient_lstm85_blinded = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'LSTM_AUC85','No')
    doctor_patient_lstm85_non_blinded = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'LSTM_AUC85','Yes')
    doctor_patient_lstm95_blinded = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'LSTM_AUC95','No')
    doctor_patient_lstm95_non_blinded = get_doctor_patient_of_diff_model(dataset, df_sample, doctor_patient_set, 'LSTM_AUC95','Yes')
    return [(doctor_patient_nomodel,'nomodel'),(doctor_patient_random,'random'),(doctor_patient_coxm_blinded,'coxm_blinded'),(doctor_patient_coxm_non_blinded,'coxm_non_blinded'),(doctor_patient_lstm75_blinded,'lstm75_blinded'),(doctor_patient_lstm75_non_blinded,'lstm75_non_blinded'),(doctor_patient_lstm85_blinded,'lstm85_blinded'),(doctor_patient_lstm85_non_blinded,'lstm85_non_blinded'),(doctor_patient_lstm95_blinded,'lstm95_blinded'),(doctor_patient_lstm95_non_blinded,'lstm95_non_blinded')]




# 按照医生年龄: 18-30  30-40  40-50  50-60
def doctor_exper4_age(doctor_patient_set, df_doctor_info):
    df_age5 = df_doctor_info[df_doctor_info['年龄'] <= 30]
    df_age10 = df_doctor_info[(df_doctor_info['年龄'] > 30) & (df_doctor_info['年龄'] <= 40)]
    df_age15 = df_doctor_info[(df_doctor_info['年龄'] > 40) & (df_doctor_info['年龄'] <= 50)]
    df_age20 = df_doctor_info[(df_doctor_info['年龄'] > 50) & (df_doctor_info['年龄'] <= 60)]

    doctor_patient_set_age5 = set()
    doctor_patient_set_age10 = set()
    doctor_patient_set_age15 = set()
    doctor_patient_set_age20 = set()
    for (doctor_id, patient_id) in doctor_patient_set:
        if doctor_id in set(df_age5[doctor_logid]):
            doctor_patient_set_age5.add((doctor_id, patient_id))
        if doctor_id in set(df_age10[doctor_logid]):
            doctor_patient_set_age10.add((doctor_id, patient_id))
        if doctor_id in set(df_age15[doctor_logid]):
            doctor_patient_set_age15.add((doctor_id, patient_id))
        if doctor_id in set(df_age20[doctor_logid]):
            doctor_patient_set_age20.add((doctor_id, patient_id))
    return doctor_patient_set_age5, doctor_patient_set_age10, doctor_patient_set_age15, doctor_patient_set_age20



def doctor_exper5_department(doctor_patient_set, df_doctor_info):
    df_1 = df_doctor_info[df_doctor_info['科室'].isin(['急诊', '急诊科', '急诊内科', '急诊医学科'])]
    df_2 = df_doctor_info[df_doctor_info['科室'].isin(['ICU', 'icu', '重症医学二病区', '重症医学科二病区', '重症医学科一病区', '重症医学科', '重症科', '重症医学', '呼吸与危重症医学科'])]
    df_3 = df_doctor_info[df_doctor_info['科室'].isin(['内科', '肾内科', '心内科', '呼吸内科', '神经内科', '心血管内科', '综合内科', '内分泌科', '呼吸'])]
    df_4 = df_doctor_info[df_doctor_info['科室'].isin(['神经外科', '普外科', '心胸血管外科', '胸心血管外科（监护室）'])]
    df_5 = df_doctor_info[df_doctor_info['科室'].isin(['骨科'])]
    df_6 = df_doctor_info[df_doctor_info['科室'].isin(['儿科'])]
    df_7 = df_doctor_info[df_doctor_info['科室'].isin(['眼科'])]
    df_8 = df_doctor_info[df_doctor_info['科室'].isin(['妇科', '妇产科'])]
    df_9 = df_doctor_info[df_doctor_info['科室'].isin(['中医科', '中医肛肠科', '中医康复科'])]
    df_10 = df_doctor_info[df_doctor_info['科室'].isin(['感染性疾病科'])]
    df_11 = df_doctor_info[df_doctor_info['科室'].isin(['风湿免疫'])]
    df_12 = df_doctor_info[df_doctor_info['科室'].isin(['脾胃科'])]
    df_13 = df_doctor_info[df_doctor_info['科室'].isin(['神经综合科'])]
    df_14 = df_doctor_info[df_doctor_info['科室'].isin(['麻醉科'])]


    set1 = set()
    set2 = set()
    set3 = set()
    set4 = set()
    set5 = set()
    set6 = set()
    set7 = set()
    set8 = set()
    set9 = set()
    set10 = set()
    set11 = set()
    set12 = set()
    set13 = set()
    set14 = set()

    for (doctor_id, patient_id) in doctor_patient_set:
        for depart_set,df_depart in [(set1,df_1),(set2,df_2),(set3,df_3),(set4,df_4),(set5,df_5),(set6,df_6),(set7,df_7),
                                     (set8,df_8),(set9,df_9),(set10,df_10),(set11,df_11),(set12,df_12),(set13,df_13),(set14,df_14)]:
            if doctor_id in set(df_depart[doctor_logid]):
                depart_set.add((doctor_id, patient_id))
    return set1,set2,set3,set4,set5,set6,set7,set8,set9,set10,set11,set12,set13,set14

def doctor_exper6_unitlevel(doctor_patient_set, df_doctor_info):
    second = {key: value for key, value in doctor_unit_dict.items() if value == 1}
    third = {key: value for key, value in doctor_unit_dict.items() if value == 2}
    univ = {key: value for key, value in doctor_unit_dict.items() if value == 3}

    df_1 = df_doctor_info[df_doctor_info['工作单位'].isin(second)]
    df_2 = df_doctor_info[df_doctor_info['工作单位'].isin(third)]
    df_3 = df_doctor_info[df_doctor_info['工作单位'].isin(univ)]
    set1 = set()
    set2 = set()
    set3 = set()
    for (doctor_id, patient_id) in doctor_patient_set:
        for depart_set,df_depart in [(set1,df_1),(set2,df_2),(set3,df_3)]:
            if doctor_id in set(df_depart[doctor_logid]):
                depart_set.add((doctor_id, patient_id))
    return set1,set2,set3


# 按照患者label和医生诊断返回：
#只考虑脓毒症患者 0h，3h，-3h的受试者
# 第一种分法：①label不是3h：诊断为一般脓毒症、严重脓毒症，如果label是3h，诊断为一般脓毒症、严重脓毒症，高度疑似，低度疑似；②其他诊断
# 第二种分法：①医生给的诊断为无脓毒症；②其他诊断
def doctor_exper4(dataset,doctor_patient_set, df_sample,df_doctor_diag,flag):
    #找出0h、3h和-3h的受试者
    df_sample_range = df_sample[df_sample['TIME_RANGE'].notna()]
    patid_range_set = set(df_sample_range['UNIQUE_ID'])
    doctor_patient_label = set()
    for doc_id,pat_id in doctor_patient_set:
        if pat_id > 20000:
            if pat_id - 20000 in patid_range_set:
                doctor_patient_label.add((doc_id,pat_id))
        else:
            if pat_id in patid_range_set:
                doctor_patient_label.add((doc_id,pat_id))

    first_column1 = set() #表示第一种分法的第一列
    first_column2 = set() #表示第一种分法的第二列
    second_column1 = set() #表示第二种分法的第一列
    second_column2 = set() #表示第二种分法的第二列

    for doc_id,pat_id in doctor_patient_label:
        if pat_id > 20000:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == pat_id-20000]
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == pat_id]

        time_range = df_sample_id.iloc[0]['TIME_RANGE']
        df_doctor_diag_id = df_doctor_diag[(df_doctor_diag['doctor_id'] == doc_id) & (df_doctor_diag['patient_id'] == pat_id)]
        if flag == 'first':
            doc_diag = get_first_diag(df_doctor_diag_id)
        else:
            doc_diag = get_final_diag(df_doctor_diag_id)

        if time_range == '3h':
            if any(keyword in doc_diag for keyword in ["一般脓毒症", "严重脓毒症", "高度疑似", "低度疑似"]):
                first_column1.add((doc_id,pat_id))
            else:
                first_column2.add((doc_id,pat_id))
        else:
            if any(keyword in doc_diag for keyword in ["一般脓毒症", "严重脓毒症"]):
                first_column1.add((doc_id,pat_id))
            else:
                first_column2.add((doc_id,pat_id))

        if '无脓毒症' in doc_diag:
            second_column1.add((doc_id, pat_id))
        else:
            second_column2.add((doc_id, pat_id))
    # print(f'诊断模型为{flag}诊断 统计如下 first_column1: {len(first_column1)} ,first_column2: {len(first_column2)} , second_column1: {len(second_column1)}, second_column2: {len(second_column2)}')

    # print(f'诊断模型为{flag}诊断 统计如下 {len(first_column1)} {len(first_column2)} {len(second_column1)} {len(second_column2)}')

    return first_column1,first_column2,second_column1,second_column2


def doctor_exper4_countnum(dataset,doc_pat_column, df_sample,model_sort, is_display):
    #找出0h、3h和-3h的受试者

    doctor_patient_temp = get_doctor_patient_of_diff_model(dataset, df_sample, doc_pat_column, model_sort, is_display) #dataset表示在在12000版的数据集上统计，column表示docpat的键值对

    subject_id_set = set()
    for doctor_patient in doctor_patient_temp:
        patient_id = doctor_patient[1]
        if patient_id > 20000:
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        subject_id = df_sample_id.iloc[0]['SUBJECT_ID']
        subject_id_set.add(subject_id)
    subject_id_num = len(subject_id_set)

    # print(subject_id_num)#患者数
    print(len(doctor_patient_temp)) #诊断样例数据




# 已经诊断的连续患者
def patient_seq(df_sample, df_sample_3000, doctor_patient_set, patient_set, df_doctor_diag, df_syslog):
    uniqueid_pats_seq = contine_patient(df_sample, df_sample_3000)
    log_patids = []
    hadm_tr_dict = {}
    for pat_seq_dict in uniqueid_pats_seq:
        for key_tr in pat_seq_dict.keys():
            uniqueid_list = pat_seq_dict.get(key_tr)
            for uniqueid in uniqueid_list:
                if uniqueid in patient_set:
                    log_patids.append(uniqueid)
                    hadm_tr_dict[uniqueid] = key_tr
                if uniqueid + 20000 in patient_set:
                    log_patids.append(uniqueid + 20000)
                    hadm_tr_dict[uniqueid + 20000] = key_tr

    df_result = pd.DataFrame()
    for (doctor_id, patient_id) in doctor_patient_set:
        if patient_id not in log_patids:
            continue

        if patient_id > 20000:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id - 20000]
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        df_doctor_diag_id = df_doctor_diag[
            (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]
        df_sys_log_id = df_syslog[(df_syslog['accountname'] == doctor_id) & (df_syslog['patient_id'] == patient_id)]

        ai_prediction = list(df_sample_id['AI模型预测结果'])[0]
        edit_count_first = first_diag_edit_counts(df_doctor_diag_id)
        edit_count_final = final_diag_edit_counts(df_doctor_diag_id)
        diag_first = get_first_diag(df_doctor_diag_id)
        diag_final = get_final_diag(df_doctor_diag_id)
        diag_acc_first = get_diag_acc(df_sample_id, diag_first)
        diag_acc_final = get_diag_acc(df_sample_id, diag_final)
        if diag_acc_final is None:
            continue
        hadmid_timerange = hadm_tr_dict.get(patient_id)
        diagtime_first = get_first_diag_time(df_sys_log_id, df_doctor_diag_id)
        diagtime_final = get_final_diag_time(df_sys_log_id, df_doctor_diag_id)
        if diagtime_final <= diagtime_first:  # 这种情况就是诊断时间跨天了
            continue
        if patient_id > 20000:
            patient_id = patient_id - 20000
        row_data = {
            'UNIQUE_ID': patient_id,
            'HADMID_TIMERANGE': hadmid_timerange,
            'AI模型预测结果': ai_prediction,
            '医生id': doctor_id,
            '医生初步诊断结果': diag_first,
            '医生最终诊断结果': diag_final,
            '医生初步诊断是否正确': diag_acc_first,
            '医生最终诊断是否正确': diag_acc_final,
            '医生初步诊断时间': diagtime_first,
            '医生最终诊断时间': diagtime_final,
            '医生初步诊断修改次数': edit_count_first,
            '医生最终诊断修改次数': edit_count_final
        }
        df_result = df_result.append(row_data, ignore_index=True)
    new_order = ['UNIQUE_ID','HADMID_TIMERANGE',	'AI模型预测结果','医生id','医生初步诊断结果','医生最终诊断结果','医生初步诊断是否正确','医生最终诊断是否正确','医生初步诊断时间','医生最终诊断时间','医生初步诊断修改次数','医生最终诊断修改次数']
    df_result = df_result[new_order]
    df_result.to_csv('D:\\4-work\\14-mimic-iv\9-系统日志\\log_analyse\\result\\seq_patient.csv', index=False,
                     encoding='gbk')


# 每个医生的患者诊断情况
def doctor_every(df_sample, doctor_set, df_doctor_diag, df_syslog):
    # 一共一百多医生
    df_result = pd.DataFrame()

    for doctor_id in doctor_set:
        patid_list = []
        diag_seq_list = []
        diag_first_list = []
        diag_final_list = []
        diag_first_acc_list = []
        diag_final_acc_list = []
        diagtime_first_list = []
        diagtime_final_list = []
        edit_diag_first_list = []
        edit_diag_final_list = []
        df_doctor = df_doctor_diag[(df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['operation'] == '新增诊断')]
        df_doctor = df_doctor.sort_values(by='time', ascending=True)
        patient_id_set = set(df_doctor['patient_id'])
        i = 0
        for pat_id in patient_id_set:
            i = i + 1
            if pat_id > 20000:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == pat_id - 20000]
            else:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == pat_id]
            df_doctor_diag_id = df_doctor_diag[
                (df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == pat_id)]
            df_sys_log_id = df_syslog[(df_syslog['accountname'] == doctor_id) & (df_syslog['patient_id'] == pat_id)]
            edit_count_first = first_diag_edit_counts(df_doctor_diag_id)
            edit_count_final = final_diag_edit_counts(df_doctor_diag_id)
            diag_first = get_first_diag(df_doctor_diag_id)
            diag_final = get_final_diag(df_doctor_diag_id)
            diag_acc_first = get_diag_acc(df_sample_id, diag_first)
            diag_acc_final = get_diag_acc(df_sample_id, diag_final)
            if diag_acc_final is None:
                continue
            diagtime_first = get_first_diag_time(df_sys_log_id, df_doctor_diag_id)
            diagtime_final = get_final_diag_time(df_sys_log_id, df_doctor_diag_id)
            if diagtime_final <= diagtime_first:  # 这种情况就是诊断时间跨天了
                continue

            patid_list.append(pat_id)
            diag_seq_list.append(i)
            diag_first_list.append(diag_first)
            diag_final_list.append(diag_final)
            diag_first_acc_list.append(diag_acc_first)
            diag_final_acc_list.append(diag_acc_final)
            diagtime_first_list.append(diagtime_first)
            diagtime_final_list.append(diagtime_final)
            edit_diag_first_list.append(edit_count_first)
            edit_diag_final_list.append(edit_count_final)

        row_data = {
            '医生id': doctor_id,
            '医生诊断的患者id': patid_list,
            '医生诊断患者的顺序': diag_seq_list,
            '医生初步诊断结果': diag_first_list,
            '医生最终诊断结果': diag_final_list,
            '医生初步诊断是否正确': diag_first_acc_list,
            '医生最终诊断是否正确': diag_final_acc_list,
            '医生初步诊断时间': diagtime_first_list,
            '医生最终诊断时间': diagtime_final_list,
            '医生初步诊断修改的次数': edit_diag_first_list,
            '医生最终诊断修改的次数': edit_diag_final_list
        }
        df_result = df_result.append(row_data, ignore_index=True)
    new_order = ['医生id','医生诊断的患者id','医生诊断患者的顺序','医生初步诊断结果','医生最终诊断结果','医生初步诊断是否正确','医生最终诊断是否正确','医生初步诊断时间','医生最终诊断时间','医生初步诊断修改的次数','医生最终诊断修改的次数']
    df_result = df_result[new_order]
    df_result.to_csv('D:\\4-work\\14-mimic-iv\9-系统日志\\log_analyse\\result\\every_doctor.csv', index=False,
                     encoding='gbk')

    # 结果保存到csv  column：医生id，医生诊断的患者id[],医生诊断患者的顺序[], 医生初步诊断结果[],医生最终诊断结果[],医生初步诊断是否正确[],医生最终诊断是否正确[],医生初步诊断时间[],医生最终诊断时间[],医生初步诊断修改的次数[],医生最终诊断修改的次数[]


# 每个医生诊断过程中诊断概率和模型概率的对比
def doctor_diag_compare(df_sample, doctor_patient_set, df_doctor_diag):
    doctor_set = set()
    patient_set = set()
    for item in doctor_patient_set:
        doctor_set.add(item[0])
        patient_set.add(item[1])
    # doctor_set = list(doctor_set)[:10]

    for doctor_id in doctor_set:
        if doctor_id != 6:
            continue

        patient_label_list = []
        model_sort_list = []
        first_diag_accept_list = []
        final_diag_accept_list = []
        model_visible_list = []
        subejectid_list = []

        df_doctor = df_doctor_diag[(df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['operation'] == '新增诊断')]
        df_doctor = df_doctor.sort_values(by='time', ascending=True)
        diag_patient_id_set = set(df_doctor['patient_id'])
        diag_patient_id_set = diag_patient_id_set & patient_set

        df_doctor_diag_group = df_doctor_diag.groupby(['doctor_id', 'patient_id'])


        for patient_id in diag_patient_id_set:
            if patient_id > 20000:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id - 20000]
            else:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            pat_label = get_label(df_sample_id.iloc[0])
            pro_0h, pro_3h, model_sort, model_visib = model_diagpro(df_sample_id.iloc[0]['AI模型预测结果'])
            subject_id = (df_sample_id.iloc[0]['SUBJECT_ID'])

            first_diag = get_first_diag(df_doctor_diag_group.get_group((doctor_id, patient_id)))
            final_diag = get_final_diag(df_doctor_diag_group.get_group((doctor_id, patient_id)))
            diag_pro_dict = {'无脓毒症': 0, '低度疑似脓毒症': 0.5, '高度疑似脓毒症': 0.75, '一般脓毒症': 1, '严重脓毒症': 1}
            first_diag_pro = None
            final_diag_pro = None
            for key in diag_pro_dict.keys():
                if key in first_diag:
                    first_diag_pro = diag_pro_dict.get(key)
                if key in final_diag:
                    final_diag_pro = diag_pro_dict.get(key)

            if all(v is not None for v in [first_diag_pro,final_diag_pro]):
                first_accept = 0
                final_accept = 0
                if pro_0h is not None and pro_0h < 0.5:
                    if first_diag_pro < 0.5:
                        first_accept = 1
                    if final_diag_pro < 0.5:
                        final_accept = 1
                else:
                    if first_diag_pro >= 0.5:
                        first_accept = 1
                    if final_diag_pro >= 0.5:
                        final_accept = 1
                # patient_label_list.append(f'{patient_id}_label{pat_label}')
                patient_label_list.append(patient_id)
                first_diag_accept_list.append(first_accept)
                final_diag_accept_list.append(final_accept)
                model_sort_list.append(model_sort)
                model_visible_list.append(model_visib)
                subejectid_list.append(subject_id)
        motivation_result_jpg(doctor_id, patient_label_list, first_diag_accept_list, final_diag_accept_list,
                              model_sort_list, model_visible_list,subejectid_list)

#目的是找出一个医生诊断一个患者 分别有模型和无模型 有模型最好准确率是95
def doctor_diag_compare2(df_sample, doctor_patient_set, df_doctor_diag,df_syslog,df_patient_check):
    doctor_set = set()
    patient_set = set()
    for item in doctor_patient_set:
        doctor_set.add(item[0])
        patient_set.add(item[1])

    for doctor_id in doctor_set:
        patient_label_list = []
        model_sort_list = []
        model_visible_list = []
        subejectid_list = []
        patient_list = []
        range_list = []

        df_doctor = df_doctor_diag[(df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['operation'] == '新增诊断')]
        df_doctor = df_doctor.sort_values(by='time', ascending=True)
        diag_patient_id_set = set(df_doctor['patient_id'])
        diag_patient_id_set = diag_patient_id_set & patient_set

        for patient_id in diag_patient_id_set:
            if patient_id > 20000:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id - 20000]
            else:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]


            pat_label,range = get_label_range(df_sample_id.iloc[0])

            pro_0h, pro_3h, model_sort, model_visib = model_diagpro(df_sample_id.iloc[0]['AI模型预测结果'])
            subject_id = (df_sample_id.iloc[0]['SUBJECT_ID'])

            patient_list.append(patient_id)
            patient_label_list.append(pat_label)
            range_list.append(range)
            model_sort_list.append(model_sort)
            model_visible_list.append(model_visib)
            subejectid_list.append(subject_id)


        counter = Counter(subejectid_list)
        duplicates_subids = [item for item, count in counter.items() if count > 1]

        for duplicates_subid in duplicates_subids:
            model_sort_temp = []
            model_visible_temp = []
            patid_temp = []
            patid_label_temp = []
            range_temp = []
            for index, subejectid in enumerate(subejectid_list):
                if subejectid == duplicates_subid:
                    model_sort_temp.append(model_sort_list[index])
                    patid_temp.append(patient_list[index])
                    patid_label_temp.append(patient_label_list[index])
                    range_temp.append(range_list[index])
                    model_visible_temp.append(model_visible_list[index])

            if None in model_sort_temp and (('LSTM_AUC95' in model_sort_temp)) :
                print(f'\n--------docid {doctor_id} ,重复患者 {duplicates_subid}--------')
                for index,(sort,patient_id,patid_label,range,visible) in enumerate(zip(model_sort_temp,patid_temp,patid_label_temp,range_temp,model_visible_temp)):
                    #计算初步诊断时间、下一步检查看了什么、最终诊断时间

                    df_doctor_diag_id = df_doctor_diag[(df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['patient_id'] == patient_id)]
                    df_sys_log_id = df_syslog[(df_syslog['accountname'] == doctor_id) & (df_syslog['patient_id'] == patient_id)]

                    df_patient_check_id = df_patient_check[(df_patient_check['doctor_id'] == doctor_id) & (df_patient_check['patient_id'] == patient_id)]
                    df_pc_next = df_patient_check_id.drop_duplicates(subset=['exam_type','imaging_type','pathogen_type','culture_smear_type'])  # 在下一步检查在日志中培养和涂片算两类
                    view_next = len(df_pc_next)
                    first_time = get_first_diag_time(df_sys_log_id,df_doctor_diag_id)
                    final_time = get_final_diag_time(df_sys_log_id,df_doctor_diag_id)

                    first_diag = get_first_diag(df_doctor_diag_id)
                    final_diag = get_final_diag(df_doctor_diag_id)

                    if patient_id > 20000:
                        df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id - 20000]
                    else:
                        df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
                    mod_diag = model_diag(df_sample_id.iloc[0]['AI模型预测结果'])

                    print(f'doctor_id {doctor_id} ,duplicates_subid {duplicates_subid},patid_label {patid_label},range {range},'
                          f'model_sort {sort},visible {visible},mod_diag {mod_diag},view_next {view_next},'
                          f'first_time {first_time},final_time {final_time},first_diag {first_diag},final_diag {final_diag}')
                    if len(df_pc_next) > 0:
                        print(df_pc_next[['exam_type','imaging_type','pathogen_type','culture_smear_type']])


#每个医生 每个模型的正负样本

def doctor_exper_13(df_sample, doctorid_set,df_doctor_diag):
    # 一共125个医生  有医生信息的是121
    # 创建DataFrame
    df_result = pd.DataFrame()

    for doctor_id in doctorid_set:
        df_doctor = df_doctor_diag[(df_doctor_diag['doctor_id'] == doctor_id) & (df_doctor_diag['operation'] == '新增诊断')]
        patient_id_set = set(df_doctor['patient_id'])
        nomodel_NO_3h = 0
        nomodel_NO_0h = 0
        nomodel_NO_3hback = 0
        nomodel_NO_normal = 0

        randommodel_NO_3h = 0
        randommodel_NO_0h = 0
        randommodel_NO_3hback = 0
        randommodel_NO_normal = 0

        trewscore_NO_3h = 0
        trewscore_NO_0h = 0
        trewscore_NO_3hback = 0
        trewscore_NO_normal = 0

        lstm75_NO_3h = 0
        lstm75_NO_0h = 0
        lstm75_NO_3hback = 0
        lstm75_NO_normal = 0

        lstm85_NO_3h = 0
        lstm85_NO_0h = 0
        lstm85_NO_3hback = 0
        lstm85_NO_normal = 0

        lstm95_NO_3h = 0
        lstm95_NO_0h = 0
        lstm95_NO_3hback = 0
        lstm95_NO_normal = 0

        trewscore_Yes_3h = 0
        trewscore_Yes_0h = 0
        trewscore_Yes_3hback = 0
        trewscore_Yes_normal = 0

        lstm75_Yes_3h = 0
        lstm75_Yes_0h = 0
        lstm75_Yes_3hback = 0
        lstm75_Yes_normal = 0

        lstm85_Yes_3h = 0
        lstm85_Yes_0h = 0
        lstm85_Yes_3hback = 0
        lstm85_Yes_normal = 0

        lstm95_Yes_3h = 0
        lstm95_Yes_0h = 0
        lstm95_Yes_3hback = 0
        lstm95_Yes_normal = 0

        patnum = len(patient_id_set)
        for patient_id in patient_id_set:
            #诊断患者的模型
            #模型是否可见
            #患者label
            if patient_id > 20000:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id - 20000]
            else:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            illtime = str(df_sample_id.iloc[0]['ILL_TIME'])
            timerange = df_sample_id.iloc[0]['TIME_RANGE']
            pro_0h, pro_3h, model_sort, model_visib = model_diagpro(df_sample_id.iloc[0]['AI模型预测结果'])

            if model_sort == 'RandomModel':
                if model_visib == 'No':
                    if illtime != 'nan':#有患病时间
                       if timerange == '3h':
                           randommodel_NO_3h +=1
                       elif timerange == '0h':
                            randommodel_NO_0h +=1
                       else:#-3h
                            randommodel_NO_3hback +=1
                    else:#正常患者
                        randommodel_NO_normal +=1

            elif model_sort == 'TREWScore':
                if model_visib == 'No':
                    if illtime != 'nan':
                        if timerange == '3h':
                            trewscore_NO_3h +=1
                        elif timerange == '0h':
                            trewscore_NO_0h +=1
                        else:  # -3h
                            trewscore_NO_3hback +=1
                    else:  # 正常患者
                        trewscore_NO_normal +=1
                else:  # 模型可见
                    if illtime != 'nan':
                        if timerange == '3h':
                            trewscore_Yes_3h +=1
                        elif timerange == '0h':
                            trewscore_Yes_0h +=1
                        else:  # -3h
                            trewscore_Yes_3hback +=1
                    else:  # 正常患者
                        trewscore_Yes_normal +=1
            elif model_sort == 'LSTM_AUC75':
                if model_visib == 'No':
                    if illtime != 'nan':
                        if timerange == '3h':
                            lstm75_NO_3h +=1
                        elif timerange == '0h':
                            lstm75_NO_0h +=1
                        else:  # -3h
                            lstm75_NO_3hback +=1
                    else:  # 正常患者
                        lstm75_NO_normal +=1
                else:  # 模型可见
                    if illtime != 'nan':
                        if timerange == '3h':
                            lstm75_Yes_3h +=1
                        elif timerange == '0h':
                            lstm75_Yes_0h +=1
                        else:  # -3h
                            lstm75_Yes_3hback +=1
                    else:  # 正常患者
                        lstm75_Yes_normal +=1
            elif model_sort == 'LSTM_AUC85':
                if model_visib == 'No':
                    if illtime != 'nan':
                        if timerange == '3h':
                            lstm85_NO_3h +=1
                        elif timerange == '0h':
                            lstm85_NO_0h +=1
                        else:  # -3h
                            lstm85_NO_3hback +=1
                    else:  # 正常患者
                        lstm85_NO_normal += 1
                else:  # 模型可见
                    if illtime != 'nan':
                        if timerange == '3h':
                            lstm85_Yes_3h +=1
                        elif timerange == '0h':
                            lstm85_Yes_0h +=1
                        else:  # -3h
                            lstm85_Yes_3hback +=1
                    else:  # 正常患者
                        lstm85_Yes_normal +=1
            elif model_sort == 'LSTM_AUC95':
                if model_visib == 'No':
                    if illtime != 'nan':
                        if timerange == '3h':
                            lstm95_NO_3h += 1
                        elif timerange == '0h':
                            lstm95_NO_0h += 1
                        else:  # -3h
                            lstm95_NO_3hback += 1
                    else:  # 正常患者
                        lstm95_NO_normal += 1
                else:  # 模型可见
                    if illtime != 'nan':
                        if timerange == '3h':
                            lstm95_Yes_3h +=1
                        elif timerange == '0h':
                            lstm95_Yes_0h +=1
                        else:  # -3h
                            lstm95_Yes_3hback +=1
                    else:  # 正常患者
                        lstm95_Yes_normal += 1
            else:#无模型
                if illtime != 'nan':
                    if timerange == '3h':
                        nomodel_NO_3h += 1
                    elif timerange == '0h':
                        nomodel_NO_0h += 1
                    else:  # -3h
                        nomodel_NO_3hback += 1
                else:  # 正常患者
                    nomodel_NO_normal += 1

        new_row_data = {
                '医生ID': doctor_id,
                '诊断患者总数':patnum,
                '无模型_不可见_3h': nomodel_NO_3h,
                '无模型_不可见_0h': nomodel_NO_0h,
                '无模型_不可见_-3h': nomodel_NO_3hback,
                '无模型_不可见_正常患者': nomodel_NO_normal,
                '随机模型_不可见_3h': randommodel_NO_3h,
                '随机模型_不可见_0h': randommodel_NO_0h,
                '随机模型_不可见_-3h': randommodel_NO_3hback,
                '随机模型_不可见_正常患者': randommodel_NO_normal,
                'TREWScore模型_不可见_3h': trewscore_NO_3h,
                'TREWScore模型_不可见_0h': trewscore_NO_0h,
                'TREWScore模型_不可见_-3h': trewscore_NO_3hback,
                'TREWScore模型_不可见_正常患者': trewscore_NO_normal,
                'LSTM_AUC75模型_不可见_3h': lstm75_NO_3h,
                'LSTM_AUC75模型_不可见_0h': lstm75_NO_0h,
                'LSTM_AUC75模型_不可见_-3h': lstm75_NO_3hback,
                'LSTM_AUC75模型_不可见_正常患者': lstm75_NO_normal,
                'LSTM_AUC85模型_不可见_3h': lstm85_NO_3h,
                'LSTM_AUC85模型_不可见_0h': lstm85_NO_0h,
                'LSTM_AUC85模型_不可见_-3h': lstm85_NO_3hback,
                'LSTM_AUC85模型_不可见_正常患者': lstm85_NO_normal,
                'LSTM_AUC95模型_不可见_3h': lstm95_NO_3h,
                'LSTM_AUC95模型_不可见_0h': lstm95_NO_0h,
                'LSTM_AUC95模型_不可见_-3h': lstm95_NO_3hback,
                'LSTM_AUC95模型_不可见_正常患者': lstm95_NO_normal,
                'TREWScore模型_可见_3h': trewscore_Yes_3h,
                'TREWScore模型_可见_0h': trewscore_Yes_0h,
                'TREWScore模型_可见_-3h': trewscore_Yes_3hback,
                'TREWScore模型_可见_正常患者': trewscore_Yes_normal,
                'LSTM_AUC75模型_可见_3h': lstm75_Yes_3h,
                'LSTM_AUC75模型_可见_0h': lstm75_Yes_0h,
                'LSTM_AUC75模型_可见_-3h': lstm75_Yes_3hback,
                'LSTM_AUC75模型_可见_正常患者': lstm75_Yes_normal,
                'LSTM_AUC85模型_可见_3h': lstm85_Yes_3h,
                'LSTM_AUC85模型_可见_0h': lstm85_Yes_0h,
                'LSTM_AUC85模型_可见_-3h': lstm85_Yes_3hback,
                'LSTM_AUC85模型_可见_正常患者': lstm85_Yes_normal,
                'LSTM_AUC95模型_可见_3h': lstm95_Yes_3h,
                'LSTM_AUC95模型_可见_0h': lstm95_Yes_0h,
                'LSTM_AUC95模型_可见_-3h': lstm95_Yes_3hback,
                'LSTM_AUC95模型_可见_正常患者': lstm95_Yes_normal
            }
        df_result = df_result.append(new_row_data, ignore_index=True)
    new_order = ['医生ID','诊断患者总数',
               '无模型_不可见_3h', '无模型_不可见_0h', '无模型_不可见_-3h', '无模型_不可见_正常患者',
               '随机模型_不可见_3h', '随机模型_不可见_0h', '随机模型_不可见_-3h', '随机模型_不可见_正常患者',
               'TREWScore模型_不可见_3h', 'TREWScore模型_不可见_0h', 'TREWScore模型_不可见_-3h', 'TREWScore模型_不可见_正常患者',
               'LSTM_AUC75模型_不可见_3h', 'LSTM_AUC75模型_不可见_0h', 'LSTM_AUC75模型_不可见_-3h', 'LSTM_AUC75模型_不可见_正常患者',
               'LSTM_AUC85模型_不可见_3h', 'LSTM_AUC85模型_不可见_0h', 'LSTM_AUC85模型_不可见_-3h', 'LSTM_AUC85模型_不可见_正常患者',
               'LSTM_AUC95模型_不可见_3h', 'LSTM_AUC95模型_不可见_0h', 'LSTM_AUC95模型_不可见_-3h', 'LSTM_AUC95模型_不可见_正常患者',
               'TREWScore模型_可见_3h', 'TREWScore模型_可见_0h', 'TREWScore模型_可见_-3h', 'TREWScore模型_可见_正常患者',
               'LSTM_AUC75模型_可见_3h', 'LSTM_AUC75模型_可见_0h', 'LSTM_AUC75模型_可见_-3h', 'LSTM_AUC75模型_可见_正常患者',
               'LSTM_AUC85模型_可见_3h', 'LSTM_AUC85模型_可见_0h', 'LSTM_AUC85模型_可见_-3h', 'LSTM_AUC85模型_可见_正常患者',
               'LSTM_AUC95模型_可见_3h', 'LSTM_AUC95模型_可见_0h', 'LSTM_AUC95模型_可见_-3h', 'LSTM_AUC95模型_可见_正常患者']
    df_result = df_result[new_order]
    df_result.to_csv('D:\\every_doctor_permodel.csv', index=False, encoding='gbk')
    print('完成文件写入')