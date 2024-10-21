import sys

path = sys.argv[0]
operation = sys.argv[1]
# path = 'D:\\4-work\\14-mimic-iv\\9-系统日志\\log_analyse_demo\\'
# operation = 'auc'

#-*- coding: utf-8 -*-
sys.path.append(path) #Modify to project path
from src.experiment import *


if __name__ == '__main__':


    root = path+'\\dataset_demo\\'

    df_syslog = pd.read_csv(root + 'sys_log.csv', encoding='gbk')

    df_patient_check = pd.read_csv(root+'patient_check.csv', encoding='gbk')

    df_doctor_diag = pd.read_csv(root+'doctor_diag.csv', encoding='gbk')

    df_sample = pd.read_csv(root+'data.csv',encoding='gbk')

    df_sample = df_sample[df_sample['TIME_RANGE'] != '-3h']
    df_sample_have_illtime = df_sample[df_sample['ILL_TIME'].notna()]
    df_sample_no_illtime = df_sample[~df_sample['ILL_TIME'].notna()]

    df_doctor_info = pd.read_csv(root+'doctor_info.csv')

    df_syslog_group = df_syslog.groupby(['accountname','patient_id'])
    df_patient_check_group = df_patient_check.groupby(['doctor_id','patient_id'])
    df_doctor_diag_group = df_doctor_diag.groupby(['doctor_id','patient_id'])

    doctor_patient_set1 = set()
    doctor_patient_set2 = set()
    for index, row_doctor_diag in df_doctor_diag_group:
        if index[0] in set(df_doctor_info['系统日志中医生ID']):
            doctor_patient_set1.add((index[0], index[1]))
    for index, row_doctor_diag in df_syslog_group:
        if index[0] in set(df_doctor_info['系统日志中医生ID']):
            doctor_patient_set2.add((index[0], index[1]))

    doctor_patient_set = doctor_patient_set1 & doctor_patient_set2

    # ③确定完成诊断的患者列表patient_set，所有的具有医生诊断的 患者id
    patient_set = set()
    doctor_set = set()
    for item in doctor_patient_set:
        doctor_set.add(item[0])
        patient_set.add(item[1])

    all_pat_subject_dict,no_model_patset,trews_model_vis_patset,random_model_novis_patset,trews_model_novis_patset,lstm75_model_vis_patset,lstm75_model_novis_patset,lstm85_model_vis_patset,lstm85_model_novis_patset,lstm95_model_vis_patset,lstm95_model_novis_patset  = model_visible_category(df_sample)

    sepsis_doctor_patient_set, notsepsis_doctor_patient_set = patient_exper1(df_sample, doctor_patient_set)

    #-------ant use acc------------
    df_ant_name = pd.read_csv(root + 'antibiotic_names.csv')
    ant_name_list = list(df_ant_name['ant_name'])

    #----------reduce time----------
    df_sample_group1 = add_detail_range(df_sample)
    # 3h is in sepsis patient
    df_sample_ill = df_sample_group1[df_sample_group1['ILL_TIME'].notna()]
    #-------------------------

    # print('========数据分析情况=============')
    for dataset in [12000]:
        #对不同模型循环  这个是放到论文附件中的表
        for doctor_patient_model,model_str in modelsort_exper(dataset, df_sample, doctor_patient_set): #TODO 很重要！！计算抗生素的时候这里只用脓毒症患者 sepsis_doctor_patient_set，其他计算用 doctor_patient_set

            # 按照患者分类: 脓毒症患者   非脓毒症患者
            sepsis_doctor_patient_set, notsepsis_doctor_patient_set = patient_exper1(df_sample, doctor_patient_model)

            # 按照医生
            df_sec, df_thir, df_univ = doctor_exper6_unitlevel(doctor_patient_model, df_doctor_info)
            doctor_patient_set_gender0, doctor_patient_set_gender1 = doctor_exper3_gender(doctor_patient_model,
                                                                                          df_doctor_info)
            doctor_patient_set_age30, doctor_patient_set_age40, doctor_patient_set_age50, doctor_patient_set_age60 = doctor_exper4_age(
                doctor_patient_model, df_doctor_info)
            doctor_patient_set_year5, doctor_patient_set_year10, doctor_patient_set_year15, doctor_patient_set_year20, doctor_patient_set_year25 = \
                doctor_exper2_year(doctor_patient_model, df_doctor_info)
            doctor_patient_set_level1, doctor_patient_set_level2, doctor_patient_set_level3, doctor_patient_set_level4 = \
                doctor_exper1_position(doctor_patient_model, df_doctor_info)

            # for flag in ['first', 'final']:
            for flag in ['final']:

                print(f'----------{model_str}---------------')
                for (doc_pat_temp,log_str) in [
                    (doctor_patient_model, 'all_clinician on all patient'),
                    (sepsis_doctor_patient_set, 'all_clinician on sepsis patient'),  # 仅包括脓毒症患者
                    (notsepsis_doctor_patient_set, 'all_clinician on not_sepsis patient'),
                    (doctor_patient_set_gender1, 'clinician_gender_male'),
                    (doctor_patient_set_gender0, 'clinician_gender_female'),
                    (df_sec,'clinician_hospital_level_2'), (df_thir,'clinician_hospital_level_3'), (df_univ,'clinician_hospital_level_1'),
                    (doctor_patient_set_age30, 'clinician_age_30'),
                    (doctor_patient_set_age40, 'clinician_age_40'),
                    (doctor_patient_set_age50, 'clinician_age_50'),
                    (doctor_patient_set_age60, 'clinician_age_60'),
                    (doctor_patient_set_year5, 'clinician_years_of_practice_<5'),
                    (doctor_patient_set_year10, 'clinician_years_of_practice_5-10'),
                    (doctor_patient_set_year15, 'clinician_years_of_practice_10-15'),
                    (doctor_patient_set_year20, 'clinician_years_of_practice_15-20'),
                    (doctor_patient_set_year25, 'clinician_years_of_practice_>25'),
                    (doctor_patient_set_level1, 'clinician_position_1'),
                    (doctor_patient_set_level2, 'clinician_position_2'),
                    (doctor_patient_set_level3, 'clinician_position_3'),
                    (doctor_patient_set_level4, 'clinician_position_4')]:
                            print(f'{log_str}:')
                            # auc
                            if operation == 'auc':
                                model_exper7_auc(flag, dataset, df_sample, doc_pat_temp, df_doctor_diag,  '', '')

                            # 灵敏度
                            if operation == 'sensitivity':
                                model_exper1(flag, dataset, df_sample_have_illtime, doc_pat_temp, df_doctor_diag, '', '')

                            # # 特异度
                            if operation == 'specificity':
                                model_exper1(flag, dataset, df_sample_no_illtime, doc_pat_temp, df_doctor_diag, '', '')

                            # # 抗生素使用正确率  只是留sepsis患者 要注释代码 修改上述注释的两个地方
                            if operation == 'antacc':
                                model_exper5_antacc(flag, dataset, df_sample, doc_pat_temp, df_doctor_diag, ant_name_list, '', '')

                            #诊断时间
                            if operation == 'diagtime':
                                model_exper2(flag, dataset, df_sample, doc_pat_temp, df_doctor_diag, df_syslog, '', '')

                            #诊断修改次数
                            if operation == 'diag_modify_times':
                                model_exper3(flag,dataset, df_sample, doc_pat_temp, df_doctor_diag,'', '')

                            #被查看的检查数量  初步
                            if operation == 'check_hisnum':
                                model_exper12_aver_hisnum(dataset, df_sample, doc_pat_temp, df_patient_check, df_syslog, '', '')

                            #被查看的检查数量  最终
                            if operation == 'check_nextnum':
                                model_exper12_aver_nextnum(dataset, df_sample, doc_pat_temp, df_patient_check, df_syslog, '', '')

