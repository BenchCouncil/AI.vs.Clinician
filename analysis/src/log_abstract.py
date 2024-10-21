import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.metrics import roc_auc_score, confusion_matrix,accuracy_score,roc_curve, auc
import numpy as np
from collections import Counter
from sklearn.utils import resample

# 设置Matplotlib以支持中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 指定字体为黑体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

#根据每个患者的诊断日志，抽取出诊断结果，诊断时间等信息

#①获取医生最终诊断结果
def get_final_diag(df_doctor_diag_id):
    df_doctor_diag_id = df_doctor_diag_id.sort_values(by='time', ascending=False)
    final_diag = df_doctor_diag_id.iloc[0]['final_diag']
    return str(final_diag).strip()

# 获取医生初步诊断结果
def get_first_diag(df_doctor_diag_id):
    df_doctor_diag_id = df_doctor_diag_id.sort_values(by='time', ascending=False)
    final_diag = df_doctor_diag_id.iloc[0]['primary_diag']
    return str(final_diag).strip()

#②获取医生最终诊断的时间
def get_final_diag_time(df_sys_log_id,df_doctor_diag_id):
    df_doc_diag_id = df_doctor_diag_id[df_doctor_diag_id['operation'] != '点击下一个患者时自动保存信息']
    df_diag_last = df_doc_diag_id.sort_values(by='time',ascending=False)
    #最终诊断的结束时间应该是包括修改了诊断的时间 考虑到有的医生可能第二天又修改了诊断结果，但这时候的诊断时间并不是连续性的,只求当天的
    endtime = pd.to_datetime(df_diag_last.iloc[0]['time_text'])

    df_sys_log_id.loc[:, 'create_time'] = pd.to_datetime(df_sys_log_id['create_time'],format='%Y/%m/%d %H:%M')
    df_sys_log_id = df_sys_log_id[df_sys_log_id['create_time'] > endtime-pd.Timedelta(hours=1)]
    df_sys_log_id = df_sys_log_id[df_sys_log_id['module'] != '注销登录']

    df_sys_last = df_sys_log_id.sort_values(by='create_time', ascending=True)
    starttime = df_sys_last.iloc[0]['create_time']

    #按照分钟算
    time_diff = (int(endtime.timestamp()) - int(starttime.timestamp()))/60
    # if time_diff < 1 or time_diff > 20:
    #     print(df_sys_log_id[['module','create_time']])
    #     print(df_doctor_diag_id['time_text'])
    #     print(f'！！诊断时间可能异常： {time_diff}分钟')
    return round(time_diff,2)


#②获取医生初步诊断的时间
def get_first_diag_time(df_sys_log_id,df_doctor_diag_id):
    df_diag_id = df_doctor_diag_id[df_doctor_diag_id['operation'] != '点击下一个患者时自动保存信息']
    df_diag_id = df_diag_id[df_diag_id['final_diag'].isnull()]
    #初步诊断的结束时间应该是包括修改了诊断的时间
    df_diag_last = df_diag_id.sort_values(by='time',ascending=False)

    if len(df_diag_last) == 0:
        openerate_set = df_doctor_diag_id['operation']
        if '修改了初步诊断' in openerate_set:
            df = df_doctor_diag_id[df_doctor_diag_id['operation'] == '修改了初步诊断']
        else:
            df = df_doctor_diag_id[df_doctor_diag_id['primary_diag'].notnull()]
        if len(df) == 0: #数据中确实存在诊断中没有初步诊断的情况，这种情况返回诊断时间10000,不纳入统计分析
            return 10000
        endtime = pd.to_datetime(df.iloc[0]['time_text'])
    else:
        endtime = pd.to_datetime(df_diag_last.iloc[0]['time_text'])
    df_sys_log_id.loc[:, 'create_time'] = pd.to_datetime(df_sys_log_id['create_time'], format="%Y/%m/%d %H:%M")
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


#获取初步诊断修改的次数
def first_diag_edit_counts(df_doctor_diag_id):
    df_diag_id = df_doctor_diag_id[df_doctor_diag_id['operation'] == '修改了初步诊断']
    #因为操作类型，新增初步诊断的时候日志 中叫新增诊断，所以不用减1
    return len(df_diag_id)


#获取最终诊断修改的次数
def final_diag_edit_counts(df_doctor_diag_id):
    df_diag_id = df_doctor_diag_id[df_doctor_diag_id['operation'] == '修改了最终诊断']
    #因为操作类型，新增最终诊断的时候日志中也叫修改了最终诊断，所以减1
    if len(df_diag_id)-1 >= 0:
        return len(df_diag_id)-1
    return 0

#获取初步诊断补液剂量
def first_diag_fluid(df_doctor_diag_id):
    df_doctor_diag_id = df_doctor_diag_id.sort_values(by='time', ascending=False)
    med_list = df_doctor_diag_id.iloc[0]['primary_med']
    if str(med_list) == 'nan':
        return None
    med_list = json.loads(med_list)
    fluid_list = []
    for med in med_list:
        fluid = med.get('diluteT')
        numbers = re.findall(r'-?\d+(?:\.\d+)?', fluid)
        if len(numbers) == 0:
            unit = med.get('unit')
            if unit == 'ml': #有的医生补液直接添加在了药品的单位中
                value = med.get('value')
                fluid_list.append(float(value))
            else:
                fluid_list.append(0)
        elif (len(numbers) == 1)&('灭菌注射用水' in med):
            fluid_list.append(int(numbers[-1]))
        elif len(numbers) == 2:
            fluid_list.append(int(numbers[-1]))

    num = sum(fluid_list)
    return num

#获取最终诊断修改的次数
def final_diag_fluid(df_doctor_diag_id):
    df_doctor_diag_id = df_doctor_diag_id.sort_values(by='time', ascending=False)
    med_list = df_doctor_diag_id.iloc[0]['final_med']
    if str(med_list) == 'nan':
        return None
    med_list = json.loads(med_list)
    fluid_list = []
    for med in med_list:
        fluid = med.get('diluteT')
        numbers = re.findall(r'-?\d+(?:\.\d+)?', fluid)
        if len(numbers) == 0:
            unit = med.get('unit')
            if unit == 'ml':  # 有的医生补液直接添加在了药品的单位中
                value = med.get('value')
                fluid_list.append(float(value))
            else:
                fluid_list.append(0)
        elif (len(numbers) == 1) & ('灭菌注射用水' in med):
            fluid_list.append(int(numbers[-1]))
        elif len(numbers) == 2:
            fluid_list.append(int(numbers[-1]))
    num = sum(fluid_list)
    return num




# ② 根据sample的label和医生的诊断 计算准确率的规则
# df_sample_id：sample中的一行，表示每个患者样本
# final_diag：表示医生的最终诊断
# 返回结果：0表示医生诊断错误，1表示医生诊断正确
def get_diag_acc(df_sample_id,final_diag):
    ill_time = df_sample_id.iloc[0]['ILL_TIME']
    #首先判断医生诊断中是否有明确的诊断，有可能下了很多诊断，但是和脓毒症没有关系，这种情况不算入计算
    if '脓毒症' not in final_diag:
        return None
    # 无患病时间，真实label是无脓毒症
    if str(ill_time) == 'nan':
        if '无脓毒症' in final_diag:
            return 1
        else:
            return 0
    else:
        range = df_sample_id.iloc[0]['TIME_RANGE']
        if range == '3h':
            #有患病时间，但是时间段是3h(患病时间之前的时间段),所以真实label是无脓毒症
            if '无脓毒症' in final_diag:
                return 1
            elif '疑似' in final_diag:
                return 1
            else:
                return 0
        else:
            #有患病时间，时间段是0h和-3h(患病时间之后的时间段)，所以真实label是脓毒症
            if '无脓毒症' in final_diag:
                return 0
            else:
                return 1

def get_diag_sen_spe(test_label, y_pred):
    tn, fp, fn, tp = confusion_matrix(test_label, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def get_patient_label(df_sample_id):
    ill_time = df_sample_id.iloc[0]['ILL_TIME']
    if str(ill_time) == 'nan':
        return 0
    else:
        range = df_sample_id.iloc[0]['TIME_RANGE']
        if range == '3h':
            return 0
        else:
            return 1


#算每个医生的acc
def get_diag_acc_perdoctor(df_sample,row_doctor_diag,diag_flag):
    per_doctor_acc_list = []

    row_group = row_doctor_diag.groupby(['patient_id'])
    for patient_id_truple, row_patient in row_group:
        patient_id = patient_id_truple[0]
        if patient_id > 20000:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id - 20000]
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]

        if diag_flag == 'first':
            diag = get_first_diag(row_patient)
        else:
            diag = get_final_diag(row_patient)
        per_patient_acc = get_diag_acc(df_sample_id, diag)
        if per_patient_acc is not None:
            per_doctor_acc_list.append(per_patient_acc)
    if len(per_doctor_acc_list) == 0:
        return None
    else:
        doctor_acc = sum(per_doctor_acc_list)/len(per_doctor_acc_list)
        return doctor_acc


#尝试将医生诊断转化为概率，然后计算acc，这种方法的计算没有更新到在线文档中
def diagpro_acc(df_sample,df_doctor_diag_part,flag):
    label_list = []
    predict_list = []

    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id', 'patient_id'])
    for index, row_doctor_diag in df_doctor_diag_group:
        patient_id = index[1]
        if patient_id > 20000: #患者id最开始是前6000，后来复制了一份，id都增加了20000，为了对应到之前的样本，所以减去20000
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        label_diag = get_label(df_sample_id.iloc[0])

        if flag == 'first':
            diag = get_first_diag(row_doctor_diag)
        else:
            diag = get_final_diag(row_doctor_diag)
        predict_pro = pro_in_diag(diag,diag_pro_dict4) #暂时用了dict4 为了不报错
        if predict_pro is not None:
            predict_list.append(predict_pro)
            label_list.append(label_diag)
    thd = 0.5
    y_pred = np.array(predict_list)
    y_pred_binary = np.where(y_pred < thd , 0, 1)
    acc = accuracy_score(label_list, y_pred_binary)
    return round(acc*100,4)


#医生查看每个患者的检查
def doctor_view_check(docter_id,patient_id_list,df_patient_check_id,fold_name):

    x_labels = patient_id_list
    y_labels = ['降钙素原', '血常规', '动脉血气分析', '影像检查', '止凝血', '病原检查', '培养', '涂片']
    y_positions = range(len(y_labels))
    # 创建一个空的二维列表表示数据
    data = [[0 for _ in x_labels] for _ in y_labels]

    i = 0
    for patient_id in patient_id_list:
        df = df_patient_check_id[df_patient_check_id['patient_id'] == patient_id]
        exam_type = set(df['exam_type'])

        for view_exam in exam_type:
            j = y_labels.index(view_exam)
            data[j][i] = 1
        i +=1
    # 创建图表，设置figsize以适应数据
    fig, ax = plt.subplots(figsize=(20, 10))  # 增加图表尺寸
    # 在每个x位置绘制竖线
    for x in range(len(x_labels)):
        ax.vlines(x, ymin=0, ymax=len(y_labels) - 1, color='lightgrey', alpha=0.7)

    # 在竖线上标记数据点
    for y, row in enumerate(data):
        for x, value in enumerate(row):
            if value == 1:
                ax.scatter(x, y, color='green', s=100)  # 增加点的大小

    # 设置横纵坐标标签
    ax.set_xticks(range(len(x_labels)))  # 设置横坐标位置
    ax.set_xticklabels(x_labels, rotation=45)  # 旋转标签以改善可读性
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels,fontsize=18)

    # 设置标题和显示图表
    plt.title(f"医生ID为{docter_id}的对不同患者查看行为分析",fontsize=24)
    # 保存图表为图片
    image_path = f'D:\\4-work\\14-mimic-iv\\9-系统日志\\log_analyse\\result\\{fold_name}\\'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    plt.savefig(image_path+f'doctorid_{docter_id}.png')
    # plt.show()



#抗生素使用是否正确，对于患者即为正确
# 返回结果：0表示抗生素使用错误，1表示抗生素使用正确
def antuse_acc(df_sample_id,ant_list,med):
    ill_time = df_sample_id.iloc[0]['ILL_TIME']

    #诊断中有没有使用抗生素
    if str(med) == 'nan':
        return None
    contains_any = any(element in med for element in ant_list)

    if str(ill_time) == 'nan':#无脓毒症患者
        if contains_any:#无脓毒症患者使用了抗生素的话，使用错误
            return 0
        else:
            return 1
    else:
        range = df_sample_id.iloc[0]['TIME_RANGE']
        if range == '3h':
            if contains_any:#这个函数不对，3h用抗生素应该算正确
                return 0
            else:
                return 1
        else:
            if contains_any:
                return 1
            else:
                return 0


# 返回结果：0表示抗生素使用错误，1表示抗生素使用正确
def antuse_acc_v2(df_sample_id,ant_list,med):
    ill_time = df_sample_id.iloc[0]['ILL_TIME']

    #诊断中有没有使用抗生素
    if str(med) == 'nan':
        return None
    contains_any = any(element in med for element in ant_list)

    if str(ill_time) == 'nan':#无脓毒症患者

        if contains_any:#无脓毒症患者使用了抗生素的话，使用错误
            return 0
        else:
            return 1

    else: #有患病时间的患者，使用了抗生素的话，表示使用正确，3h（患病前）使用抗生素 也是使用正确
        timerange = df_sample_id.iloc[0]['TIME_RANGE']
        if timerange == '3h':
            return 1 # 3h用抗生素 正确，不用抗生素也正确，因为3h患病前的label是0
        if contains_any:
            return 1
        else:
            return 0



def get_label(df_sample_id):
    ill_time = df_sample_id['ILL_TIME']
    # 无患病时间，真实label是无脓毒症
    if str(ill_time) == 'nan':
        return 0
    else:
        range = df_sample_id['TIME_RANGE']
        # 有患病时间，时间段是3h,真实label是无脓毒症
        if range == '3h':
            return 0
        else:
            # 有患病时间，时间段是0h和-3h(-3h表示患病时间之后的时间段),真实label是脓毒症患者
            return 1

def get_label_range(df_sample_id):
    ill_time = df_sample_id['ILL_TIME']
    # 无患病时间，真实label是无脓毒症
    if str(ill_time) == 'nan':
        return 0,None
    else:
        range = df_sample_id['TIME_RANGE']
        # 有患病时间，时间段是3h,表示在患病时间之前的时间段，真实label是无脓毒症
        if range == '3h':
            return 0,range
        else:
            # 有患病时间，时间段是0h和-3h(-3h表示患病时间之后的时间段),真实label是脓毒症患者
            return 1,range


diag_pro_dict1 = {'无脓毒症':0,'高度疑似脓毒症_0':0.2, '低度疑似脓毒症_0':0.1, '低度疑似脓毒症_1':0.4, '高度疑似脓毒症_1':0.6, '一般脓毒症':1, '严重脓毒症':1 }
diag_pro_dict2 = {'无脓毒症':0,'高度疑似脓毒症_0':0.25, '低度疑似脓毒症_0':0.15, '低度疑似脓毒症_1':0.5, '高度疑似脓毒症_1':0.7, '一般脓毒症':1, '严重脓毒症':1 }
diag_pro_dict3 = {'无脓毒症':0,'高度疑似脓毒症_0':0.3, '低度疑似脓毒症_0':0.2, '低度疑似脓毒症_1':0.55, '高度疑似脓毒症_1':0.75,'一般脓毒症':1, '严重脓毒症':1 }
diag_pro_dict4 = {'无脓毒症':0,'高度疑似脓毒症_0':0.35, '低度疑似脓毒症_0':0.25, '低度疑似脓毒症_1':0.6, '高度疑似脓毒症_1':0.8, '一般脓毒症':1, '严重脓毒症':1 }
diag_pro_dict5 = {'无脓毒症':0,'高度疑似脓毒症_0':0.4, '低度疑似脓毒症_0':0.3, '低度疑似脓毒症_1':0.65, '高度疑似脓毒症_1':0.85, '一般脓毒症':1, '严重脓毒症':1 }

def pro_in_diag(label_diag,input_string,diag_pro_dict):
    last_found_key = None
    for key in diag_pro_dict.keys():
        key_temp = key.replace(f'_{label_diag}','')
        if key_temp in input_string:
            last_found_key = key
    return diag_pro_dict.get(last_found_key) if last_found_key else None

#取每个医生最大的auc计算平均
def diag_avg_auc(df_sample,df_doctor_diag_part,flag):
    df_doctor_group = df_doctor_diag_part.groupby('doctor_id')
    best_auc_list = []
    for index, df in df_doctor_group:
        best_auc = diag_perdoct_auc(df_sample, df, flag)
        if best_auc is not None:
            best_auc_list.append(best_auc)
    if len(best_auc_list) == 0:
        avg_auc = 'No diagnostic sample'
        auc_ci = None
    else:
        avg_auc = round(sum(best_auc_list)/len(best_auc_list)/100,4) #/100
        auc_ci = cal_ci(best_auc_list)
    return avg_auc,auc_ci

def cal_ci(value_list):
    n_bootstraps = 5000
    np.random.seed(42)
    bootstrapped = []
    for _ in range(n_bootstraps):
        sample = resample(value_list)
        bootstrapped.append(round((sum(sample) / len(sample))/100, 4)) #这里auc需要/100,4   灵敏度*100 ，2都可调节
    confidence_level = 95
    lower = round(np.percentile(bootstrapped, (100 - confidence_level) / 2),4)
    upper = round(np.percentile(bootstrapped, 100 - (100 - confidence_level) / 2),4)
    return [lower, upper]

#每个医生的auc的计算
def diag_perdoct_auc(df_sample,df_doctor_diag_part,flag):
    label_list = []
    diag_list = []
    acc_list = []
    doctor_id = None
    #按每个医生的维度，算给出每个医生疑似按40%，50%，55% 60 65这样算。然后取这个医生auc最大的情况，然后再算平均
    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id', 'patient_id'])
    for index, row_doctor_diag in df_doctor_diag_group:
        doctor_id = index[0]

        patient_id = index[1]
        if patient_id > 20000: #患者id最开始是前6000，后来复制了一份，id都增加了20000，为了对应到之前的样本，所以减去20000
            patient_id = patient_id - 20000
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        else:
            df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
        label_diag = get_label(df_sample_id.iloc[0])

        if flag == 'first':
            diag = get_first_diag(row_doctor_diag)
        else:
            diag = get_final_diag(row_doctor_diag)
        predict_pro = pro_in_diag(label_diag,diag,diag_pro_dict4)#这里只是确保后续转换成诊断概率的时候和label的list一致

        if predict_pro is not None:
            diag_list.append(diag)
            label_list.append(label_diag)
        acc_list.append(get_diag_acc(df_sample_id, diag))
    #如果label中只有一个类别  不能计算auc
    if all(elem == label_list[0] for elem in label_list) :
        return None
    diag_pro_list = [diag_pro_dict1,diag_pro_dict2,diag_pro_dict3,diag_pro_dict4,diag_pro_dict5]
    auc_list = []
    predict_list_jpg = []
    for diag_pro_dict in diag_pro_list:
        predict_list = []
        for index,diag in enumerate(diag_list):
            predict_list.append(pro_in_diag(label_list[index],diag,diag_pro_dict))
        auc = roc_auc_score(label_list, predict_list)
        predict_list_jpg.append(predict_list)
        auc_list.append(round(auc*100,2)) #计算auc 不想要输出*100 这里改成 round(auc,2)

    filtered_acc_list = [acc for acc in acc_list if acc is not None]
    avg_acc = sum(filtered_acc_list)/len(filtered_acc_list)
    # auc_jpg(doctor_id,len(label_list),round(avg_acc,2),label_list, predict_list_jpg)
    return max(auc_list)



def auc_jpg(doctor_id,pat_num,avg_acc,label_list, pred_list):
    fpr_1, tpr_1, _ = roc_curve(label_list, pred_list[0])
    fpr_2, tpr_2, _ = roc_curve(label_list, pred_list[1])
    fpr_3, tpr_3, _ = roc_curve(label_list, pred_list[2])
    fpr_4, tpr_4, _ = roc_curve(label_list, pred_list[3])
    fpr_5, tpr_5, _ = roc_curve(label_list, pred_list[4])

    # 计算 AUC
    auc_1 = auc(fpr_1, tpr_1)
    auc_2 = auc(fpr_2, tpr_2)
    auc_3 = auc(fpr_3, tpr_3)
    auc_4 = auc(fpr_4, tpr_4)
    auc_5 = auc(fpr_5, tpr_5)

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_1, tpr_1, label=f'AUC 1 = {auc_1:.2f}', color='blue')
    plt.plot(fpr_2, tpr_2, label=f'AUC 2 = {auc_2:.2f}', color='red')
    plt.plot(fpr_3, tpr_3, label=f'AUC 3 = {auc_3:.2f}', color='green')
    plt.plot(fpr_4, tpr_4, label=f'AUC 4 = {auc_4:.2f}', color='orange')
    plt.plot(fpr_5, tpr_5, label=f'AUC 5 = {auc_5:.2f}', color='purple')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'医生id为{doctor_id},诊断患者数量为{pat_num},准确率{avg_acc} 的ROC Curve')
    plt.legend()
    plt.show()

#诊断时间修改率
def diag_modify_rate(df_doctor_diag_part,flag):
    modify_list = []

    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id', 'patient_id'])
    for index, row_doctor_diag in df_doctor_diag_group:
        if flag == 'first':
            modify_num = first_diag_edit_counts(row_doctor_diag)
        else:
            modify_num = final_diag_edit_counts(row_doctor_diag)
        #如果修改次数大于0，则添加1代表有修改
        if modify_num == 0:
            modify_list.append(0)
        else:
            modify_list.append(1)
    if len(modify_list) == 0:
        modify_rate = 'No diagnostic sample'
    else:
        modify_rate = round((sum(modify_list)/len(modify_list))*100,2)
    return modify_rate

#初步诊断到最终诊断的变化统计
#列举了21种变化情况：1.有变化、2.无脓毒症à低度疑似脓毒症、3.无脓毒症à高度疑似脓毒症、4.无脓毒症à一般脓毒症、5.无脓毒症à严重脓毒症
#6.低度疑似脓毒症à无脓毒症 7.低度疑似脓毒症à高度疑似脓毒症 8.低度疑似脓毒症à一般脓毒症 9.低度疑似脓毒症à严重脓毒症 10.高度疑似脓毒症à无脓毒症
#11.高度疑似脓毒症à低度疑似脓毒症 12.高度疑似脓毒症à一般脓毒症 13.高度疑似脓毒症à严重脓毒症 14.般脓毒症à无脓毒症 15.一般脓毒症à低度疑似脓毒症
#16.一般脓毒症à高度疑似脓毒症 17.一般脓毒症à严重脓毒症 18.严重脓毒症à无脓毒症 19.严重脓毒症à低度疑似脓毒症 20.严重脓毒症à高度疑似脓毒症
#21.严重脓毒症à一般脓毒症
def diag_change(df_doctor_diag_part):
    label_list = ['无脓毒症','低度疑似脓毒症','高度疑似脓毒症','一般脓毒症','严重脓毒症']
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0
    count9 = 0
    count10 = 0
    count11 = 0
    count12 = 0
    count13 = 0
    count14 = 0
    count15 = 0
    count16 = 0
    count17 = 0
    count18 = 0
    count19 = 0
    count20 = 0
    count21 = 0

    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id', 'patient_id'])
    for index, row_doctor_diag in df_doctor_diag_group:
        first_diag = get_first_diag(row_doctor_diag)
        final_diag = get_final_diag(row_doctor_diag)

        first_diag_list = [diag for diag in label_list if diag in first_diag]
        final_diag_list = [diag for diag in label_list if diag in final_diag]
        if len(first_diag_list) == 1 and len(final_diag_list)==1:
            first = first_diag_list[0]
            final = final_diag_list[0]
            if first != final:
                count1 += 1
            if first == '无脓毒症' and final == '低度疑似脓毒症':
                count2 +=1
            if first == '无脓毒症' and final == '高度疑似脓毒症':
                count3 +=1
            if first == '无脓毒症' and final == '一般脓毒症':
                count4 +=1
            if first == '无脓毒症' and final == '严重脓毒症':
                count5 +=1
            if first == '低度疑似脓毒症' and final == '无脓毒症':
                count6 +=1
            if first == '低度疑似脓毒症' and final == '高度疑似脓毒症':
                count7 +=1
            if first == '低度疑似脓毒症' and final == '一般脓毒症':
                count8 +=1
            if first == '低度疑似脓毒症' and final == '严重脓毒症':
                count9 +=1
            if first == '高度疑似脓毒症' and final == '无脓毒症':
                count10 +=1
            if first == '高度疑似脓毒症' and final == '低度疑似脓毒症':
                count11 +=1
            if first == '高度疑似脓毒症' and final == '一般脓毒症':
                count12 +=1
            if first == '高度疑似脓毒症' and final == '严重脓毒症':
                count13 +=1
            if first == '一般脓毒症' and final == '无脓毒症':
                count14 +=1
            if first == '一般脓毒症' and final == '低度疑似脓毒症':
                count15 +=1
            if first == '一般脓毒症' and final == '高度疑似脓毒症':
                count16 +=1
            if first == '一般脓毒症' and final == '严重脓毒症':
                count17 +=1
            if first == '严重脓毒症' and final == '无脓毒症':
                count18 +=1
            if first == '严重脓毒症' and final == '低度疑似脓毒症':
                count19 +=1
            if first == '严重脓毒症' and final == '高度疑似脓毒症':
                count20 +=1
            if first == '严重脓毒症' and final == '一般脓毒症':
                count21 +=1
    change_list = [count1,count2,count3,count4,count5,count6,count7,count8,count9,count10,count11,count12,count13,count14,count15,count16,count17,count18,count19,count20,count21]
    return change_list


def diag_change_acc(df_sample,df_doctor_diag_part):
    label_list = ['无脓毒症','低度疑似脓毒症','高度疑似脓毒症','一般脓毒症','严重脓毒症']
    diag_change_acc_list = []

    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id', 'patient_id'])
    for index, row_doctor_diag in df_doctor_diag_group:
        first_diag = get_first_diag(row_doctor_diag)
        final_diag = get_final_diag(row_doctor_diag)

        first_diag_list = [diag for diag in label_list if diag in first_diag]
        final_diag_list = [diag for diag in label_list if diag in final_diag]
        if len(first_diag_list) == 1 and len(final_diag_list)==1:
            first = first_diag_list[0]
            final = final_diag_list[0]
            if first != final: #diag is change
                doctor_id = index[0]
                patient_id = index[1]
                if patient_id > 20000:  # 患者id最开始是前6000，后来复制了一份，id都增加了20000，为了对应到之前的样本，所以减去20000
                    patient_id = patient_id - 20000
                    df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
                else:
                    df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
                illtime = df_sample_id.iloc[0]['ILL_TIME']

                if str(illtime) != 'nan':# sepsis patient
                    #first diag is non-sepsis and final diag is sepsis ,this change is corrent
                    if first == '无脓毒症' and final in ['低度疑似脓毒症','高度疑似脓毒症','一般脓毒症','严重脓毒症']:
                        diag_change_acc_list.append(1)
                    else:
                        diag_change_acc_list.append(0)
                else:#non-sepsis patient
                    if first in ['低度疑似脓毒症', '高度疑似脓毒症', '一般脓毒症','严重脓毒症'] and final == '无脓毒症':
                        diag_change_acc_list.append(1)
                    else:
                        diag_change_acc_list.append(0)

    if len(diag_change_acc_list) == 0:
        diag_change_avge_acc = 'No diagnostic sample'
    else:
        diag_change_avge_acc = round((sum(diag_change_acc_list)/len(diag_change_acc_list))*100, 2)
    return diag_change_avge_acc,sum(diag_change_acc_list)



#初步到最终抗生素的使用变化
#1.用了到没用   2.没用到用了
def antibio_change(df_doctor_diag_part,all_antname_list):
    count1 = 0
    count2 = 0

    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id', 'patient_id'])
    for index, row_doctor_diag in df_doctor_diag_group:
        df_doctor_diag_id = row_doctor_diag.sort_values(by='time', ascending=False)
        first_med_list = df_doctor_diag_id.iloc[0]['primary_med']
        final_med_list = df_doctor_diag_id.iloc[0]['final_med']
        if str(first_med_list) != 'nan' and str(final_med_list) != 'nan':
            first_antibio = any(element in first_med_list for element in all_antname_list)
            final_antibio = any(element in final_med_list for element in all_antname_list)
            #用了到没用
            if first_antibio :
                if not final_antibio:
                    count1+=1
            #没用到用了
            if not first_antibio :
                if final_antibio:
                    count2+=1
    return [count1,count2]

def antibio_change_acc(df_sample,df_doctor_diag_part,all_antname_list):
    antibio_change_acc_list = []

    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id', 'patient_id'])
    for index, row_doctor_diag in df_doctor_diag_group:
        df_doctor_diag_id = row_doctor_diag.sort_values(by='time', ascending=False)
        first_med_list = df_doctor_diag_id.iloc[0]['primary_med']
        final_med_list = df_doctor_diag_id.iloc[0]['final_med']
        if str(first_med_list) != 'nan' and str(final_med_list) != 'nan':
            first_antibio = any(element in first_med_list for element in all_antname_list)
            final_antibio = any(element in final_med_list for element in all_antname_list)
            doctor_id = index[0]
            patient_id = index[1]
            if patient_id > 20000:  # 患者id最开始是前6000，后来复制了一份，id都增加了20000，为了对应到之前的样本，所以减去20000
                patient_id = patient_id - 20000
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            else:
                df_sample_id = df_sample[df_sample['UNIQUE_ID'] == patient_id]
            illtime = df_sample_id.iloc[0]['ILL_TIME']

            #用了到没用
            if first_antibio:
                if not final_antibio:
                    if str(illtime) != 'nan':  # sepsis patient
                        antibio_change_acc_list.append(0)
                    else:
                        antibio_change_acc_list.append(1)
            #没用到用了
            if not first_antibio :
                if final_antibio:
                    if str(illtime) != 'nan':  # sepsis patient
                        antibio_change_acc_list.append(1)
                    else:
                        antibio_change_acc_list.append(0)

    if len(antibio_change_acc_list) == 0:
        antibio_change_avge_acc = 'No diagnostic sample'
    else:
        antibio_change_avge_acc = round((sum(antibio_change_acc_list) / len(antibio_change_acc_list)) * 100, 2)
    return antibio_change_avge_acc, sum(antibio_change_acc_list)

#初步到最终补液的变化
def fluid_change(df_doctor_diag_part):
    up_fluid_list = []
    down_fluid_list = []
    df_doctor_diag_group = df_doctor_diag_part.groupby(['doctor_id', 'patient_id'])
    for index, row_doctor_diag in df_doctor_diag_group:
        first_fluid = first_diag_fluid(row_doctor_diag)
        final_fluid = final_diag_fluid(row_doctor_diag)
        if first_fluid is not None and final_fluid is not None:
            #初步到最终 增加
            if first_fluid < final_fluid:
                change = final_fluid-first_fluid
                up_fluid_list.append(change)
            #初步到最终 减少
            if final_fluid < first_fluid:
                change = first_fluid - final_fluid
                down_fluid_list.append(change)
    if len(up_fluid_list) == 0:
        aver_up = 'No diagnostic sample'
    else:
        aver_up = round(sum(up_fluid_list)/len(up_fluid_list),2)
    if len(down_fluid_list) == 0:
        aver_down = 'No diagnostic sample'
    else:
        aver_down = round(sum(down_fluid_list) / len(down_fluid_list),2)
    return aver_up,aver_down



#患者被查看的检查数量（历史检查、下一步检查）
def view_check_num(doctor_patient_set_part,df_syslog_part,df_patient_check_part):
    df_syslog_group = df_syslog_part.groupby(['accountname','patient_id'])
    df_patient_check_group = df_patient_check_part.groupby(['doctor_id','patient_id'])
    view_his_list = []
    view_next_list = []
    for doct_id,pat_id in doctor_patient_set_part:
        df_sl = df_syslog_group.get_group((doct_id,pat_id))
        df_sl_his = df_sl[df_sl['module'] == '历史检查']
        df_sl_his = df_sl_his[df_sl_his['exception'].str.contains('历史用药')==False]
        df_sl_his = df_sl_his.drop_duplicates(subset=['exception']) #去掉历史用药
        view_his = len(df_sl_his)
        if (doct_id, pat_id) in df_patient_check_group.groups.keys():
            df_pc_next = df_patient_check_group.get_group((doct_id, pat_id))
            df_pc_next = df_pc_next.drop_duplicates(subset=['check_type']) #在下一步检查在日志中培养和涂片算两类
            view_next = len(df_pc_next)
        else:
            view_next = 0
        view_his_list.append(view_his)
        view_next_list.append(view_next)
    if len(view_his_list) == 0:
        avg_his = 'No diagnostic sample'
        value_ci_his = None
    else:
        avg_his = round(sum(view_his_list)/len(view_his_list),2)
        value_ci_his = cal_ci(view_his_list)
    if len(view_next_list) == 0:
        avg_next = 'No diagnostic sample'
        value_ci_next =None
    else:
        avg_next = round(sum(view_next_list)/len(view_next_list),2)
        value_ci_next = cal_ci(view_next_list)
    return avg_his,avg_next,value_ci_his,value_ci_next

#绘制医生诊断的过程 纵坐标为医生诊断的概率和模型诊断的概率 横坐标为按顺序诊断的患者label
def doctor_diag_jpg(doctor_id,patient_label_list,first_diag_accept_list,final_diag_accept_list,model_sort_list,model_visible_list):
    def convert_accept_rate(lst):
        result = []
        for i in range(1, len(lst) + 1):
            current_avg = sum(lst[:i]) / i
            result.append(round(current_avg, 2))
        return result

    fig, axs = plt.subplots(4, 5, figsize=(40, 30))

    visibles = ['Yes','No']
    model_sorts = ['RandomModel','LSTM_AUC75','LSTM_AUC85','LSTM_AUC95','TREWScore']
    i = 0
    for model_sort in model_sorts:
        for visib in visibles:
            patient_label_list_temp = []
            first_diag_accept_list_temp = []
            final_diag_accept_list_temp = []
            for index, (model_sort_item, model_visible_item) in enumerate(zip(model_sort_list, model_visible_list)):
                if model_sort_item == model_sort and model_visible_item ==visib:
                    patient_label_list_temp.append(patient_label_list[index])
                    first_diag_accept_list_temp.append(first_diag_accept_list[index])
                    final_diag_accept_list_temp.append(final_diag_accept_list[index])
            j = 0
            for y_values in [convert_accept_rate(first_diag_accept_list_temp),convert_accept_rate(final_diag_accept_list_temp)]:
                if j ==0:
                    diag = '初步诊断'
                else:
                    diag = '最终诊断'
                row = i // 5
                col = i % 5
                axs[row, col].plot(patient_label_list_temp, y_values, marker='o')
                axs[row, col].set_title(f'医生id{doctor_id} {diag},模型{model_sort},可见性{visib}的诊断分析',color='blue',fontsize=18)
                axs[row, col].set_xlabel('顺序诊断的患者')
                axs[row, col].set_ylabel('医生诊断患者是对模型结果的接受率')
                axs[row, col].set_ylim(0, 1)

                i = i+1
                j = j+1

    plt.tight_layout()
    # plt.show()

    # 保存图表为图片
    image_path = f'D:\\4-work\\14-mimic-iv\\9-系统日志\\0-虚拟医生文档\\1-找例子\\模型0h和医生诊断\\'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    plt.savefig(image_path+f'doctorid_{doctor_id}.jpg')


#上面找到了doctorid为6 三甲医生 夏炳杰 主任医师
def motivation_result_jpg(doctor_id,patient_label_list,first_diag_accept_list,final_diag_accept_list,model_sort_list,model_visible_list,subejectid_list):
    def convert_accept_rate(lst):
        result = []
        for i in range(1, len(lst) + 1):
            current_avg = sum(lst[:i]) / i
            result.append(round(current_avg, 2))
        return result
    if doctor_id != 6:
        return None

    visibles = ['Yes','No']
    model_sorts = ['LSTM_AUC75','LSTM_AUC95']
    not_have_model = []
    for model_sort in model_sorts:
        for visib in visibles:
            if str(visib) == 'No' and str(model_sort) == 'LSTM_AUC75':
                continue

            patient_label_list_temp = []
            first_diag_accept_list_temp = []
            final_diag_accept_list_temp = []
            for index, (model_sort_item, model_visible_item) in enumerate(zip(model_sort_list, model_visible_list)):
                if model_sort_item == model_sort and model_visible_item ==visib:
                    patient_label_list_temp.append(patient_label_list[index])
                    first_diag_accept_list_temp.append(first_diag_accept_list[index])
                    final_diag_accept_list_temp.append(final_diag_accept_list[index])
            y_values = convert_accept_rate(final_diag_accept_list_temp)

            print(y_values)
            print(patient_label_list_temp)
            print(model_sort)
            print(visib)
    #无模型患者
    for index, (model_sort_item, model_visible_item) in enumerate(zip(model_sort_list, model_visible_list)):
        if model_sort_item is None and model_visible_item is None:
            not_have_model.append(patient_label_list[index])
    print(not_have_model)
