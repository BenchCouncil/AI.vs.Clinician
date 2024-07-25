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
import os

#The paths of all the files used are as follows
PATH = '/home/ddcui/'
ROOT = f'{PATH}/hai-med-database/mimic-coxphm/data/2-cph-newfill-final/'

df = pd.read_csv('train.csv')
dfL1 = df.pop('ill_label')
dfT1 = df.pop('period_end_to_illtime')

dfL1.to_csv("testL-sample.csv", header=False, index=False)
dfT1.to_csv("testT-sample.csv", header=False, index=False)
df.to_csv("testF-sample.csv", header=True, index=False)

#df = pd.read_csv('hfc-sample.csv')
dfL = pd.read_csv('testL-sample.csv', header=None)
dfT = pd.read_csv('testT-sample.csv', header=None)
dfF = pd.read_csv('testF-sample.csv')
print(sys.argv[1])
time = float(sys.argv[1])
dfL_arr = dfL.T.values[0]
dfT_arr = dfT.T.values[0]
print(f"before label: {dfL_arr[:50]}")
print(f"before time: {dfT_arr[:50]}")
####################
if time == 0.01 or time == '0.01':
    for i in range (0,len(dfL_arr)):
        if dfL_arr[i] == 1 and dfT_arr[i] <= 0.01:
            dfL_arr[i] = 1
        else:
            dfL_arr[i] = 0
elif time == 3 or time == '3':
    for i in range(0,len(dfL_arr)):
        if dfL_arr[i] == 1 and dfT_arr[i] < 4: # and dfT_arr[i] != 0.01:
            dfL_arr[i] = 1
        else:
            dfL_arr[i] = 0
else:
    for i in range(0,len(dfL_arr)):
        if dfL_arr[i] == 1 and dfT_arr[i] < 15: # and dfT_arr[i] > 3:
            dfL_arr[i] = 1
        else:
            dfL_arr[i] = 0
######################
print(f"after label: {dfL_arr[:50]}")
print(collections.Counter(dfL_arr))

with open('cox-cph.pickle', 'rb') as f:
    cph_load = pickle.load(f)
    result = cph_load.predict_survival_function(dfF,times=[time]) 
    result = result.applymap(lambda x: 1-x)
    arr = result.values[0]
    print(f"arr: {arr[:50]}")
    np.savetxt('testout.txt', (dfL_arr,dfT_arr,arr), delimiter=",")
    metric = roc_auc_score(dfL_arr,arr)
    print(f"auc: {metric}")
    fpr, tpr, thresholds = roc_curve(dfL_arr, arr)  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
    sens = 1-fpr
    np.savetxt('metric.txt', (sens,tpr,thresholds), delimiter=",")

    thresh_prob = 0
    for i in range(len(tpr)):
        if tpr[i] >= 0.8:
            print(f"tpr0.8: {tpr[i]}, thresh: {thresholds[i]}")
            thresh_prob = thresholds[i]
            break
        else:
            continue
    print(thresh_prob)

    num1r = 0
    num0r = 0
    num1w =0
    num0w = 0
    prob = 0
    prob_arr = []
    #j = 0
    for i in range(0,len(arr)):
        if dfL_arr[i] == 1 and arr[i] >= thresh_prob:
            num1r = num1r + 1
            prob_arr.append(arr[i]) #prob_arr[j]=arr[i]
            #j = j+1
        if dfL_arr[i] == 1 and arr[i] < thresh_prob:
            num1w = num1w + 1
            #prob_arr.append(arr[i])
        if dfL_arr[i] == 0 and arr[i] >= thresh_prob:
            num0w = num0w + 1
            #prob_arr.append(arr[i])
        if dfL_arr[i] == 0 and arr[i] < thresh_prob:
            num0r =num0r + 1
            #prob_arr.append(arr[i])
    for i in range(0,len(arr)):
        if arr[i] > 1:
            prob = prob + 1
    print(f"label 1, model right num: {num1r}")
    print(f"label 1, model wrong num: {num1w}")
    print(f"label 0, model right num: {num0r}")
    print(f"label 0, model wrong num: {num0w}")
    print(f"prob >1 num: {prob}")
    print(f"prob_arr: {len(prob_arr)}")
    
    arr_temp = np.sort(prob_arr)
    arr_95 = arr_temp[round(len(arr_temp)*0.95)]
    print(f"arr_95: {arr_95}")
    df_thresh = pd.DataFrame({'thresh_prob':[thresh_prob],'arr_95':[arr_95]})
    df_thresh.to_csv(f'thresh_prob{time}.csv')


    roc_auc = auc(fpr, tpr)  # 求auc面积
    print(f"auc1: {roc_auc}")
    os.remove("testL-sample.csv")
    os.remove("testF-sample.csv")
    os.remove("testT-sample.csv")
