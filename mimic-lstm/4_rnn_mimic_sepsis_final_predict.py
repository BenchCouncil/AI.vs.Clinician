import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

import gc
from time import time
import os
import math
import pickle
import numpy as np
import pandas as pd
from pad_sequences import PadSequences
from attention_function import attention_3d_block as Attention
from keras import backend as K
# from keras.models import Model, Input, load_model #model_from_json
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Masking, Flatten, Embedding, Dense, LSTM, TimeDistributed
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve
from keras import regularizers
from keras import optimizers
# from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization,Bidirectional

import sys




ROOT = "./mimic_database/mapped_elements/"




def get_label(df_sample, time_steps=336):
    df_label = df_sample['ill_label']
    df_label_array = df_label.values
    df_label_array = df_label_array.reshape(int(df_label_array.shape[0] / time_steps), time_steps)
    df_label_array = df_label_array[:, 0]
    array_label = []
    for i in range(len(df_label_array)):
        array_label.append(int(df_label_array[i]))
    if 'ill_label' in list(df_sample.columns):
        del df_sample['ill_label']

    sample_array = df_sample.values
    sample_array = sample_array.reshape(int(sample_array.shape[0] / time_steps), time_steps, sample_array.shape[1])

    sample_test_x = sample_array
    sample_test_y = np.array(array_label)
    return sample_test_x, sample_test_y


def get_sample_id_and_time(df_sample,column, time_steps=336):
    df_label = df_sample[column]
    df_label_array = df_label.values
    df_label_array = df_label_array.reshape(int(df_label_array.shape[0] / time_steps), time_steps)
    df_label_array = df_label_array[:, 0]
    column_y = []
    for i in range(len(df_label_array)):
        column_y.append(df_label_array[i])
    if column in list(df_sample.columns):
        del df_sample[column]

    # column_y = np.array(column_y)
    return column_y,df_sample

def get_score(test_y, predict_test_y):
    print('Confusion Matrix')
    print(confusion_matrix(test_y, np.around(predict_test_y)))
    print('Accuracy')
    print(accuracy_score(test_y, np.around(predict_test_y)))
    print('ROC AUC SCORE')
    print(roc_auc_score(test_y, predict_test_y))
    print('CLASSIFICATION REPORT')
    print(classification_report(test_y, np.around(predict_test_y)))

    count_1 = sum(x == 1 and y == 1 for x, y in zip(np.around(predict_test_y), test_y))
    count_2 = sum(x == 0 and y == 0 for x, y in zip(np.around(predict_test_y), test_y))
    count_3 = sum(x == 1 and y == 0 for x, y in zip(np.around(predict_test_y), test_y))
    count_4 = sum(x == 0 and y == 1 for x, y in zip(np.around(predict_test_y), test_y))
    print(f'real label 1,model predict 1  number {count_1}')
    print(f'real label 0,model predict 0 number {count_2}')
    print(f'real label 1,model predict 0 number {count_4}')
    print(f'real label 0,model predict 1 number {count_3}')
    print(f'sensitivity {count_1 / np.sum(test_y == 1)}')
    print(f'specificity {count_2 / np.sum(test_y == 0)}')

    fpr, tpr, thresholds = roc_curve(test_y, predict_test_y)
    return fpr, tpr, thresholds


def get_0h_3h_score():
    df_3000_sample = pd.read_csv('mimic_database/mapped_elements_3000/mimic_3000_sample.csv',
                                 usecols=['HADM_ID', 'START_ENDTIME', 'ILL_TIME'])
    df_predict_0 = pd.read_csv('mimic_database/mapped_elements_3000/mimic_3000_sample_0h_predict_data.csv')
    df_predict_3 = pd.read_csv('mimic_database/mapped_elements_3000/mimic_3000_sample_3h_predict_data.csv')

    df_3000_sample = pd.merge(df_3000_sample, df_predict_0, on=['HADM_ID', 'START_ENDTIME'], how='inner')
    df_3000_sample = pd.merge(df_3000_sample, df_predict_3, on=['HADM_ID', 'START_ENDTIME'], how='inner')

    real_label_list = []
    predict_list = []
    for index, row in df_3000_sample.iterrows():
        predict_0h = float(row['0h_PREDICT'])
        predict_3h = float(row['3h_PREDICT'])
        ill_time = row['ILL_TIME']
        if str(ill_time) == 'nan':
            real_label = 0
        else:
            real_label = 1
        # 首先判断0h是否发病
        if predict_0h <= 0.5:
            if predict_3h <= 0.5:
                pre = min(predict_0h, predict_0h)
            else:
                pre = predict_3h
        else:
            pre = predict_0h
        predict_list.append(pre)
        real_label_list.append(real_label)

    predict_test_y = np.array(predict_list)
    test_y = np.array(real_label_list)
    print('============= SAMPLE ===============')
    get_score(test_y, predict_test_y)


def predict_with_tscore_same(test_y,predict_test_y,predict_sample_test_y):

    fpr, tpr, thresholds = get_score(test_y, predict_test_y)
    thresh_prob = 0
    for i in range(len(tpr)):
        if tpr[i] >= 0.8:
            print(f"tpr0.8: {tpr[i]}, thresh: {thresholds[i]}")
            thresh_prob = thresholds[i]
            break
        else:
            continue
    num1r = 0
    num0r = 0
    num1w = 0
    num0w = 0
    prob = 0
    prob_arr = []
    # j = 0
    for i in range(0, len(predict_test_y)):
        if test_y[i] == 1 and predict_test_y[i] >= thresh_prob:
            num1r = num1r + 1
            prob_arr.append(predict_test_y[i])  # prob_arr[j]=arr[i]
            # j = j+1
        if test_y[i] == 1 and predict_test_y[i] < thresh_prob:
            num1w = num1w + 1
        if test_y[i] == 0 and predict_test_y[i] >= thresh_prob:
            num0w = num0w + 1
        if test_y[i] == 0 and predict_test_y[i] < thresh_prob:
            num0r = num0r + 1
    for i in range(0, len(predict_test_y)):
        if predict_test_y[i] > 1:
            prob = prob + 1
    # print(f"label 1, model right num: {num1r}")
    # print(f"label 1, model wrong num: {num1w}")
    # print(f"label 0, model right num: {num0r}")
    # print(f"label 0, model wrong num: {num0w}")
    # print(f"prob >1 num: {prob}")
    # print(f"prob_arr: {len(prob_arr)}")

    arr_temp = np.sort(prob_arr)
    # print(arr_temp)
    arr_95 = arr_temp[round(len(arr_temp) * 0.95)]
    print(f"arr_95: {arr_95}")

    print('===============SAMPLE===============')

    for i in range(0, len(predict_sample_test_y)):
        if predict_sample_test_y[i] <= thresh_prob:
            predict_sample_test_y[i] = predict_sample_test_y[i] * 0.5 / thresh_prob
        elif predict_sample_test_y[i] > thresh_prob and predict_sample_test_y[i] < arr_95:
            predict_sample_test_y[i] = (predict_sample_test_y[i] - thresh_prob) * 0.5 / (arr_95 - thresh_prob) + 0.5
        else:
            predict_sample_test_y[i] = 1
    # print(f"arr: {predict_sample_test_y[:50]}")
    return predict_sample_test_y

def to_csv(hadm_id_list,start_time_list,end_time_list,predict_0h_95,predict_0h_85,predict_0h_75,predict_3h_95,predict_3h_85,predict_3h_75):

    df_sample_result = pd.DataFrame(
        {'hadm_id': hadm_id_list, 'start_time': start_time_list,'end_time':end_time_list,
         'predict_0h_95': predict_0h_95.tolist(),'predict_0h_85': predict_0h_85.tolist(),'predict_0h_75': predict_0h_75.tolist(),
         'predict_3h_95': predict_3h_95.tolist(),'predict_3h_85': predict_3h_85.tolist(),'predict_3h_75': predict_3h_75.tolist(),})
    df_sample_result.to_csv('mimiciv_3000_sample_predict_result.csv',index=False)


def sample_predict_final():
    ROOT = f'/home/ddcui//hai-med-database/mimic-lstm/data/'
    data_path = ROOT + f'diff_meanvalue_del_missdata_0.4/'

    model_0h_95 = load_model('model_0h/saved_models_0h_95/lstm_3layer_model_0h_epochs50_diff_meanvalue_del_missdata_0.4.h5')
    model_0h_85 = load_model('model_0h/saved_models_0h_85/lstm_1layer_model_0h_epochs3_diff_meanvalue_del_missdata_0.4.h5')
    model_0h_75 = load_model('model_0h/saved_models_0h_75/lstm_1layer_model_0h_epochs2_diff_meanvalue_del_missdata_0.4.h5')
    model_3h_95 = load_model('model_3h/saved_models_3h_95/lstm_3layer_model_3h_epochs50_diff_meanvalue_del_missdata_0.4.h5')
    model_3h_85 = load_model('model_3h/saved_models_3h_85/lstm_1layer_model_3h_epochs3_diff_meanvalue_del_missdata_0.4.h5')
    model_3h_75 = load_model('model_3h/saved_models_3h_75/lstm_1layer_model_3h_epochs2_diff_meanvalue_del_missdata_0.4.h5')

    df_test_3h_iii = pd.read_csv(data_path + 'lstm_3h_test_model_input_data_by_diff_meanvalue_del_missdata_0.4.csv')
    df_test_3h_iv = pd.read_csv(data_path + 'lstm_3h_mimiciv_test_model_input_data_by_diff_meanvalue.csv')
    df_test_3h = pd.concat([df_test_3h_iii, df_test_3h_iv])
    test_3h_x, test_3h_y = get_label(df_test_3h, time_steps=336)

    df_test_0h_iii = pd.read_csv(data_path + 'lstm_0h_test_model_input_data_by_diff_meanvalue_del_missdata_0.4.csv')
    df_test_0h_iv = pd.read_csv(data_path + 'lstm_0h_mimiciv_test_model_input_data_by_diff_meanvalue.csv')
    df_test_0h = pd.concat([df_test_0h_iii, df_test_0h_iv])
    test_0h_x, test_0h_y = get_label(df_test_0h, time_steps=336)

    df_sample = pd.read_csv(data_path + 'lstm_mimiciv_sample_model_input_data_by_diff_meanvalue_predict.csv')

    #先取出hadm_id start_time end_time（选sample的开始时间和结束时间） group_id
    hadm_id_list,df_sample = get_sample_id_and_time(df_sample, 'hadm_id')
    start_time_list,df_sample = get_sample_id_and_time(df_sample, 'start_time')
    end_time_list,df_sample = get_sample_id_and_time(df_sample, 'end_time')

    sample_test_x, sample_test_y = get_label(df_sample, time_steps=336)


    predict_test_0h_95_y = model_0h_95.predict(test_0h_x)
    predict_test_0h_85_y = model_0h_85.predict(test_0h_x)
    predict_test_0h_75_y = model_0h_75.predict(test_0h_x)
    predict_test_3h_95_y = model_3h_95.predict(test_3h_x)
    predict_test_3h_85_y = model_3h_85.predict(test_3h_x)
    predict_test_3h_75_y = model_3h_75.predict(test_3h_x)

    predict_sample_test_0h_95_y = model_0h_95.predict(sample_test_x)
    predict_sample_test_0h_85_y = model_0h_85.predict(sample_test_x)
    predict_sample_test_0h_75_y = model_0h_75.predict(sample_test_x)
    predict_sample_test_3h_95_y = model_3h_95.predict(sample_test_x)
    predict_sample_test_3h_85_y = model_3h_85.predict(sample_test_x)
    predict_sample_test_3h_75_y = model_3h_75.predict(sample_test_x)

    predict_0h_95 = predict_with_tscore_same(test_0h_y, predict_test_0h_95_y, predict_sample_test_0h_95_y)
    predict_0h_85 = predict_with_tscore_same(test_0h_y, predict_test_0h_85_y, predict_sample_test_0h_85_y)
    predict_0h_75 = predict_with_tscore_same(test_0h_y, predict_test_0h_75_y, predict_sample_test_0h_75_y)
    predict_3h_95 = predict_with_tscore_same(test_3h_y, predict_test_3h_95_y, predict_sample_test_3h_95_y)
    predict_3h_85 = predict_with_tscore_same(test_3h_y, predict_test_3h_85_y, predict_sample_test_3h_85_y)
    predict_3h_75 = predict_with_tscore_same(test_3h_y, predict_test_3h_75_y, predict_sample_test_3h_75_y)


    to_csv(hadm_id_list,start_time_list,end_time_list,predict_0h_95,predict_0h_85,predict_0h_75,predict_3h_95,predict_3h_85,predict_3h_75)


if __name__ == "__main__":
    sample_predict_final()
    print('end')
