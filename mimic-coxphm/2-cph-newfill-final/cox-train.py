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
import random
from lifelines.utils import k_fold_cross_validation

#The paths of all the files used are as follows
PATH = '/home/ddcui/'
ROOT = f'{PATH}/hai-med-database/mimic-coxphm/data/2-cph-newfill-final/'

df_train = pd.read_csv('train.csv')
#df_train = df_train.dropna(axis=0)
print(f"df train: {len(df_train)}")
#df_train.drop(columns=['chronic airway obstruction', 'chronic kidney disease', 'chronic liver disease', 'chronic pulmonary', 'immunodeficiency', 'organ insufficiency', 'rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)

#df_train.drop(columns=['chronic airway obstruction', 'chronic kidney disease', 'chronic liver disease', 'chronic pulmonary', 'immunodeficiency', 'organ insufficiency', 'rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase', 'rass'], inplace=True)

#df_train.drop(columns=['chronic airway obstruction', 'chronic bronchitis', 'chronic kidney disease', 'chronic liver disease', 'chronic pulmonary', 'immunodeficiency', 'organ insufficiency', 'rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)
#df_train.drop(columns=['rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase', 'pao2/fio2 ratio'], inplace=True)
#df_train.drop(columns=['rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)

df_train.drop(columns=['rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)

print(f"df_train before: {df_train.shape[1]}")

#df_train.drop(columns=['subject_id', 'hadm_id', 'start_time', 'end_time'], inplace=True)


#df_train = df.sample(frac=0.7,random_state=2023)
#df_val = df.drop(index = df_train.index)
#df_testL = df_val.pop('ill_label')
#df_testT = df_val.pop('period_end_to_illtime')
#df_testL.to_csv("testL-cph.csv", header=False, index=False)
#df_testT.to_csv("testT-cph.csv", header=False, index=False)
#df_val.to_csv("testF-cph.csv", header=True, index=False)

num0 = 0
for i in df_train.index:
    if df_train['period_end_to_illtime'][i] >= 336:
        num0 = num0 + 1
print(f"num0: {num0}")

print(f"df_train: {df_train.shape[1]}")

print(f"训练集大小：{len(df_train)}")
print("*"*20)
#统计失效事件和删失数据的数量
print(df_train['ill_label'].value_counts())
###
cph = CoxPHFitter(l1_ratio=1,penalizer=0.05)
cph.fit(df=df_train, duration_col='period_end_to_illtime', event_col='ill_label', show_progress=True,fit_options={'step_size':0.1,'precision':1e-08,'max_steps':1000})
cph.print_summary()
###
with open('cox-cph.pickle', 'wb') as f:
    pickle.dump(cph, f) # saving my trained cph model as my.pickle
