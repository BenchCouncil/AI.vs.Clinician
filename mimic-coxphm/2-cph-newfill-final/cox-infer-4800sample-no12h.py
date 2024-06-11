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

#thresh0 = pd.read_csv(f'thresh_prob0.01.csv')
#thresh_prob0 = thresh0['thresh_prob'][0]
#arr_95_0 = thresh0['arr_95'][0]

#thresh3 = pd.read_csv(f'thresh_prob3.csv')
#thresh_prob3 = thresh3['thresh_prob'][0]
#arr_95_3 = thresh3['arr_95'][0]

#thresh12 = pd.read_csv(f'thresh_prob3.csv')
#thresh_prob12 = thresh12['thresh_prob'][0]
#arr_95_12 = thresh12['arr_95'][0]


#print(f"thresh_prob0: {thresh_prob0}, arr_95_0: {arr_95_0}, thresh_prob3: {thresh_prob3}, arr_95_3: {arr_95_3}")

dfread = pd.read_csv('sample.csv') #('newsample3000.csv')
#dfread.drop(columns=['chronic airway obstruction', 'chronic kidney disease', 'chronic liver disease', 'chronic pulmonary', 'immunodeficiency', 'organ insufficiency', 'rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)
#dfread.drop(columns=['rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)
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

#df = pd.read_csv('hfc-sample.csv')
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
######################
#print(f"after label: {dfL_arr[:50]}")
#print(collections.Counter(dfL_arr))

#df['ill_label'] = df['ill_label']#.astype(bool)
#df['chronic airway obstruction'] = df['chronic airway obstruction'].astype(bool)

#print(df.loc[events, 'diabetes'].var())
#print(df.loc[~events, 'diabetes'].var())
#df = df.dropna(axis=0)
#df.drop(columns=['chronic airway obstruction', 'chronic bronchitis', 'chronic kidney disease', 'chronic liver disease', 'chronic pulmonary', 'diabetes', 'emphysema', 'heart failure', 'immunodeficiency', 'organ insufficiency', 'rena insufficiency', 'biliary complaint', 'cardiac complaint', 'dementia complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)

#df_train = df.sample(frac=0.8,random_state=2022)
#dfnotime = df_train.drop('period_end_to_illtime', axis=1)
#dftime = df_train.pop('period_end_to_illtime')
#df_val = df.drop(index = df_train.index)
#df_testL = df_val.pop('ill_label')
#df_testT = df_val.pop('period_end_to_illtime')
#df_testF = df_val.drop(columns=['period_end_to_illtime'], inplace=False)
#print(f"df_train: {df_train.shape[1]}")
#print(f"df_val: {df_val.shape[1]}")
#print(f"df_testF: {df_testF.shape[1]}")

#print(f"训练集大小：{len(df_train)}")
#print(f"验证集大小：{len(df_val)}")

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
    #dfL_time12_array = label_with_time(12)
    arr_prob0 = infer_with_time(0.01,dfL_time0_array)
    arr_prob3 = infer_with_time(3,dfL_time3_array)
    #arr_prob12 = infer_with_time(12,dfL_time12_array)
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
        #resultpd = pd.DataFrame(result,columns=['predict_result']) 
        #print(resultpd)
        df_idinfo.reset_index(drop=True, inplace=True)
        #print(df_idinfo)
        df_idinfo.insert(df_idinfo.shape[1],'predict_result',result)
        #df_idinfo = pd.concat([df_idinfo,resultpd], axis=1, sort=False)
        #print(f"df_idinfo: {df_idinfo}")
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
        #resultpd = pd.DataFrame(result,columns=['predict_result'])
        #print(resultpd)
        df_idinfo.reset_index(drop=True, inplace=True)
        #print(df_idinfo)
        df_idinfo.insert(df_idinfo.shape[1],'predict_result',result)
        #pd.merge(df_idinfo,resultpd,left_index=True,right_index=True)
        #df_idinfo = pd.concat([df_idinfo,resultpd], axis=1, sort=False)
        #print(f"df_idinfo: {df_idinfo}")
    df_idinfo.to_csv(f'predict_result_no12h_group{group}.csv', header=True, index=False, encoding='utf_8_sig', index_label=False)
    
#with open('cox-cph.pickle', 'rb') as f:
#    cph_load = pickle.load(f)
#    result = cph_load.predict_survival_function(dfF,times=[time]) 
#    result = result.applymap(lambda x: 1-x)
#    arr = result.values[0]
#    print(f"arr: {arr[:50]}")
#    np.savetxt('testout.txt', (dfL_arr,dfT_arr,arr), delimiter=",")

#    metric = roc_auc_score(dfL_arr,arr)
##    print(f"auc: {metric}")
#    fpr, tpr, thresholds = roc_curve(dfL_arr, arr)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
#    sens = 1-fpr
#    np.savetxt(f'metric{time}.txt', (sens,tpr,thresholds), delimiter=",")
#    roc_auc = auc(fpr, tpr)  # 求auc面积
#    print(f"auc1: {roc_auc}")

#    num1r = 0
#    num0r = 0
#    num1w =0
#    num0w = 0
#    prob = 0
#    for i in range(0,len(arr)):
#        if dfL_arr[i] == 1 and arr[i] > thresh_prob:
#            num1r = num1r + 1
#        if dfL_arr[i] == 1 and arr[i] <= thresh_prob:
#            num1w = num1w + 1
#        if dfL_arr[i] == 0 and arr[i] > thresh_prob:
#            num0w = num0w + 1
#        if dfL_arr[i] == 0 and arr[i] <thresh_prob:
#            num0r =num0r + 1
#    for i in range(0,len(arr)):
#        if arr[i] > 1:
#            prob = prob + 1
#    print(f"label 1, model right num: {num1r}")
#    print(f"label 1, model wrong num: {num1w}")
#    print(f"label 0, model right num: {num0r}")
#    print(f"label 0, model wrong num: {num0w}")
#    print(f"prob >1 num: {prob}")

#    for i in range(0,len(arr)):
#        if arr[i] <= thresh_prob:
#            arr[i] = arr[i]*0.5/thresh_prob
#        elif arr[i] > thresh_prob and arr[i] < arr_95:
#            arr[i] = (arr[i] - thresh_prob)*0.5/(arr_95 - thresh_prob)+0.5
#        else:
#            arr[i] = 1
#    print(f"arr: {arr[:50]}")
#    plt.plot(fpr, tpr, linewidth=1, color="red", label='AD ROC (AUC = {0:.4f})'.format(roc_auc))
#    plt.savefig('auc.jpg')
    #    else:
     #       continue
#cph_new.print_summary() # should produce the same output as cph.summary

#print(cph.predict_partial_hazard(df_val))
