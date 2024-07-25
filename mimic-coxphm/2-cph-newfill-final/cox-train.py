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
print(f"df train: {len(df_train)}")

df_train.drop(columns=['rena insufficiency', 'cardiac complaint', 'fall complaint', 'gastrointestinal bleed complaint', 'seizure complaint', 'stroke complain', 'aspartate aminotransferase'], inplace=True)
print(f"df_train before: {df_train.shape[1]}")


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
