import ast
import re
import pandas as pd


root = ".//dataset//"

# ①获取模型的诊断结果
def model_diag(text):
    if str(text) == 'nan':
        return None
    if 'TREWScore' in str(text):
        text = re.sub(r":(\w+),", r":'\1',", text, count=2)
    data_dict = ast.literal_eval(text)
    # 获取第一个键及其对应的值
    first_key = list(data_dict.keys())[0]
    model_diag = data_dict[first_key]
    return model_diag

def model_diagpro(text):
    if str(text) == 'nan':
        return None,None,None,None
    if 'TREWScore' in str(text):
        text = re.sub(r":(\w+),", r":'\1',", text, count=2)
    data_dict = ast.literal_eval(text)
    # 获取第一个键及其对应的值
    last_key = list(data_dict.keys())[-1]
    model_predictpro = data_dict[last_key]
    split_parts = model_predictpro.split(',')
    pro_0h = split_parts[0].split(':')[-1]
    pro_3h = split_parts[1].split(':')[-1]
    model_sort = list(data_dict.keys())[0]
    model_visib = data_dict[list(data_dict.keys())[1]]
    return float(pro_0h),float(pro_3h),model_sort,model_visib






#返回定义好的四类：无模型、TREWS 显示模型、Random 不显示模型、TREWS 不显示模型
def model_visible_category(df_sample):

    no_model_patset = []
    trews_model_vis_patset = []
    random_model_novis_patset = []
    trews_model_novis_patset = []


    lstm75_model_vis_patset = []
    lstm75_model_novis_patset = []
    lstm85_model_vis_patset = []
    lstm85_model_novis_patset = []
    lstm95_model_vis_patset = []
    lstm95_model_novis_patset = []


    all_pat_subject_dict = {}#key AI模型预测结果，value:SUBJECT_ID
    for index, row in df_sample.iterrows():
        pat_id = row['UNIQUE_ID']
        subject_id = row['SUBJECT_ID']
        model_pre = row['AI模型预测结果']
        group = row['GROUP']
        all_pat_subject_dict[pat_id] = subject_id

        if str(model_pre) == 'nan':
            no_model_patset.append(pat_id)
        else:
            if 'TREWScore' in model_pre and 'Yes' in model_pre:
                trews_model_vis_patset.append(pat_id)
            if 'RandomModel' in model_pre and 'No' in model_pre:
                random_model_novis_patset.append(pat_id)
            if ('TREWScore' in model_pre) and ('No' in model_pre) and group==1:
                trews_model_novis_patset.append(pat_id)

            if 'LSTM_AUC75' in model_pre and 'Yes' in model_pre:
                lstm75_model_vis_patset.append(pat_id)
            if ('LSTM_AUC75' in model_pre) and ('No' in model_pre) and group==1:
                lstm75_model_novis_patset.append(pat_id)

            if 'LSTM_AUC85' in model_pre and 'Yes' in model_pre:
                lstm85_model_vis_patset.append(pat_id)
            if ('LSTM_AUC85' in model_pre) and ('No' in model_pre) and group==1:
                lstm85_model_novis_patset.append(pat_id)

            if 'LSTM_AUC95' in model_pre and 'Yes' in model_pre:
                lstm95_model_vis_patset.append(pat_id)
            if ('LSTM_AUC95' in model_pre) and ('No' in model_pre) and group==1:
                lstm95_model_novis_patset.append(pat_id)

    return all_pat_subject_dict,no_model_patset,trews_model_vis_patset,random_model_novis_patset,trews_model_novis_patset,lstm75_model_vis_patset,lstm75_model_novis_patset,lstm85_model_vis_patset,lstm85_model_novis_patset,lstm95_model_vis_patset,lstm95_model_novis_patset



#⑥ 获取连续时间段的同一患者
def contine_patient(df_sample,df):
    #dataset中有这两个文件
    seq_pat_list = []
    df_group = df.groupby(['subject_id','hadm_id'])
    for index,row in df_group:
        if len(row) == 3:
            subject_id = index[0]
            hadm_id = index[1]
            df_sample_id = df_sample[(df_sample['SUBJECT_ID'] == subject_id)&(df_sample['HADM_ID'] == hadm_id) ]

            df_sample_3h_before = df_sample_id[df_sample_id['TIME_RANGE'] == '-3h']
            before_3h_uniqueid_list = list(df_sample_3h_before['UNIQUE_ID'])

            df_sample_0h = df_sample_id[df_sample_id['TIME_RANGE'] == '0h']
            curr_3h_uniqueid_list = list(df_sample_0h['UNIQUE_ID'])

            df_sample_after_3h = df_sample_id[df_sample_id['TIME_RANGE'] == '3h']
            after_3h_uniqueid_list = list(df_sample_after_3h['UNIQUE_ID'])
            pat_dict = {}

            pat_dict[f"{hadm_id}_-3h"] = before_3h_uniqueid_list
            pat_dict[f"{hadm_id}_0h"] = curr_3h_uniqueid_list
            pat_dict[f"{hadm_id}_3h"] = after_3h_uniqueid_list
            seq_pat_list.append(pat_dict)
    return seq_pat_list



def add_detail_range(df_sample):
    for index,row in df_sample.iterrows():
        admittime = row['ADMITTIME']
        illtime = row['ILL_TIME']
        if str(illtime) == 'nan':
            continue
        ill_starttime,ill_endtime = get_illtime_from_admittime(admittime,illtime)

        illtime_1 = ill_starttime - pd.Timedelta(hours=3)
        illtime_2 = ill_starttime - pd.Timedelta(hours=2)
        illtime_3 = ill_starttime - pd.Timedelta(hours=1)

        illtime_4 = ill_endtime + pd.Timedelta(hours=1)
        illtime_5 = ill_endtime + pd.Timedelta(hours=2)
        illtime_6 = ill_endtime + pd.Timedelta(hours=3)

        currenttime = row['START_ENDTIME']
        current_starttime = pd.to_datetime(currenttime.split('~')[0])
        current_timerange = row['TIME_RANGE']

        if current_timerange == '0h':
            df_sample.at[index,'time_range_detail'] = '0h'
        elif current_timerange == '3h':#患病前
            if current_starttime == illtime_1:
                df_sample.at[index, 'time_range_detail'] = '3h'
            elif current_starttime== illtime_2:
                df_sample.at[index, 'time_range_detail'] = '2h'
            elif current_starttime== illtime_3:
                df_sample.at[index, 'time_range_detail'] = '1h'

        elif current_timerange == '-3h':
            if current_starttime == ill_endtime:
                df_sample.at[index, 'time_range_detail'] = '-1h'
            elif current_starttime == illtime_4 :
                df_sample.at[index, 'time_range_detail'] = '-2h'
            elif current_starttime == illtime_5 :
                df_sample.at[index, 'time_range_detail'] = '-3h'

    return df_sample

def get_illtime_from_admittime(admittime,ill_time):
    ill_time = pd.to_datetime(ill_time)
    admittime = pd.to_datetime(admittime)
    for i in range(336):
        starttime = admittime + pd.Timedelta(hours=i+1)
        endtime = starttime+ pd.Timedelta(hours=1)
        if ill_time>= starttime and ill_time<= endtime:
            return starttime,endtime
    return None,None

