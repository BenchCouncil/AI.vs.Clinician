'''cox model for sepsis'''

import numpy as np
import pandas as pd
import pickle
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.utils.sklearn_adapter import sklearn_adapter
#from lifelines import CoxTimeVaryingFitter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import auc,roc_curve,roc_auc_score
import sys
import collections
import matplotlib.pyplot as plt

print(sys.argv[1])
group = float(sys.argv[1])

#The paths of all the files used are as follows
PATH = '/home/ddcui/'
ROOT = f'{PATH}/hai-med-database/mimic-coxphm/data/2-cph-newfill-final/'

dfread = pd.read_csv('sample.csv') #('newsample3000.csv')
dfread.drop(columns=['rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)

df = dfread[dfread['group_id'] == group]
print(df.columns)
df_idinfo = df[['subject_id', 'hadm_id', 'ill_label', 'period_end_to_illtime', 'start_time', 'end_time', 'group_id']]
df.drop(columns=['subject_id', 'hadm_id', 'start_time', 'end_time', 'group_id'], inplace=True)

dfL1 = df.pop('ill_label')
dfT1 = df.pop('period_end_to_illtime')

dfL1.to_csv(f"testL-sample.csv", header=False, index=False)
dfT1.to_csv(f"testT-sample.csv", header=False, index=False)
df.to_csv(f"testF-sample.csv", header=True, index=False)

dfL = pd.read_csv(f'testL-sample.csv', header=None)
dfT = pd.read_csv(f'testT-sample.csv', header=None)
dfF = pd.read_csv(f'testF-sample.csv')
dfL_arr = dfL.T.values[0]
dfT_arr = dfT.T.values[0]
print(f"before label: {dfL_arr[:50]}")
print(f"before time: {dfT_arr[:50]}")

def label_with_time(time):
    ####################
    dfL_time_arr=[0] * len(dfL_arr)
    if time == 0.01 or time == '0.01':
        for i in range (0,len(dfL_arr)):
            if dfL_arr[i] == 1 and dfT_arr[i] <= 0.01:
                dfL_time_arr[i] = 1
            else:
                dfL_time_arr[i] = 0
    elif time == 3 or time == '3':
        for i in range(0,len(dfL_arr)):
            if dfL_arr[i] == 1 and dfT_arr[i] < 4: # and dfT_arr[i] != 0.01:
                dfL_time_arr[i] = 1
            else:
                dfL_time_arr[i] = 0
    else:
        for i in range(0,len(dfL_arr)):
            if dfL_arr[i] == 1 and dfT_arr[i] < 15: # and dfT_arr[i] > 3:
                dfL_time_arr[i] = 1
            else:
                dfL_time_arr[i] = 0
    print(f"after label: {dfL_time_arr[:50]}")
    return dfL_time_arr


def infer_with_time(time,dfL_time_arr):
    thresh = pd.read_csv(f'thresh_prob{time}.csv')
    thresh_prob = thresh['thresh_prob'][0]
    arr_95 = thresh['arr_95'][0]
    print(f"thresh_prob: {thresh_prob}, arr_95: {arr_95}")

    with open('cox-cph.pickle', 'rb') as f:
        cph_load = pickle.load(f)
        result = cph_load.predict_survival_function(dfF,times=[time])
        result = result.applymap(lambda x: 1-x)
        arr = result.values[0]
        metric = roc_auc_score(dfL_time_arr,arr)
        print(f"auc: {metric}")
        fpr, tpr, thresholds = roc_curve(dfL_time_arr, arr)
        sens = 1-fpr
        np.savetxt(f'metric{time}.txt', (sens,tpr,thresholds), delimiter=",")
        roc_auc = auc(fpr, tpr) 
        print(f"auc1: {roc_auc}")
        num1r = 0
        num0r = 0
        num1w =0
        num0w = 0
        for i in range(0,len(arr)):
            if dfL_time_arr[i] == 1 and arr[i] > thresh_prob:
                num1r = num1r + 1
            if dfL_time_arr[i] == 1 and arr[i] <= thresh_prob:
                num1w = num1w + 1
            if dfL_time_arr[i] == 0 and arr[i] > thresh_prob:
                num0w = num0w + 1
            if dfL_time_arr[i] == 0 and arr[i] <thresh_prob:
                num0r =num0r + 1
        print(f"label 1, model right num: {num1r}")
        print(f"label 1, model wrong num: {num1w}")
        print(f"label 0, model right num: {num0r}")
        print(f"label 0, model wrong num: {num0w}")
        for i in range(0,len(arr)):
            if arr[i] <= thresh_prob:
                arr[i] = arr[i]*0.5/thresh_prob
            elif arr[i] > thresh_prob and arr[i] < arr_95:
                arr[i] = (arr[i] - thresh_prob)*0.5/(arr_95 - thresh_prob)+0.5
            else:
                arr[i] = 1
        print(f"arr: {arr[:50]}")
    return arr


if __name__ == '__main__':

    dfL_time0_array = label_with_time(0.01)
    dfL_time3_array = label_with_time(3)

    arr_prob0 = infer_with_time(0.01,dfL_time0_array)
    arr_prob3 = infer_with_time(3,dfL_time3_array)

    state = []
    result = []
    if group == 1:
        visible = 'No'
        for i in range(0,len(arr_prob0)):
            if max(arr_prob0[i], arr_prob3[i]) < 0.5:
                state.append('正常')
            elif arr_prob0[i] >= 0.5: #== max(arr_prob0[i], arr_prob3[i], arr_prob12[i]): # >= arr_prob3[i] and arr_prob0[i] >= arr_prob12[i]:
                state.append('脓毒症')
            else:
                state.append('脓毒症预警') 
        
        for i in range(0,len(arr_prob0)):
             result.append("{'TREWScore':" + state[i] + "," + " 'TREWScore_IsVisible':No" + "," + " 'TREWScore_Predict_Time':" + " '预测脓毒症概率0h:" + str(format(arr_prob0[i],'.3f')) + "," + "3h:" + str(format(arr_prob3[i],'.3f'))  + "'}")


        df_idinfo.reset_index(drop=True, inplace=True)
        df_idinfo.insert(df_idinfo.shape[1],'predict_result',result)

    if group == 2:
        visible = 'Yes'
        for i in range(0,len(arr_prob0)):
            if max(arr_prob0[i], arr_prob3[i]) < 0.5:
                state.append('正常')
            elif arr_prob0[i] >= 0.5: # >= arr_prob3[i] and arr_prob0[i] >= arr_prob12[i]:
                state.append('脓毒症')
            else:
                state.append('脓毒症预警')

        for i in range(0,len(arr_prob0)):
             result.append("{'TREWScore':" + state[i] + "," + " 'TREWScore_IsVisible':Yes" + "," + " 'TREWScore_Predict_Time':" + " '预测脓毒症概率0h:" + str(format(arr_prob0[i],'.3f')) + "," + "3h:" + str(format(arr_prob3[i],'.3f')) + "'}")

        df_idinfo.reset_index(drop=True, inplace=True)
        df_idinfo.insert(df_idinfo.shape[1],'predict_result',result)

    df_idinfo.to_csv(f'predict_result_no12h_group{group}.csv', header=True, index=False, encoding='utf_8_sig', index_label=False)
