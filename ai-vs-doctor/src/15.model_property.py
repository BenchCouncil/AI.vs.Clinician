import pandas as pd
from src.constant import *


columns = ['MODEL_ID', 'MODEL_NAME','TIME_PERIOD', 'DATASET', 'SENSITIVITY', 'SPECIFICITY', 'PRECISION', 'AUC']

data_dict = {
    'RandomModel-0h-TestSet-SENSITIVITY':0.5,
    'RandomModel-0h-TestSet-SPECIFICITY': 0.5,
    'RandomModel-0h-TestSet-PRECISION': 0.5,
    'RandomModel-0h-TestSet-AUC': 0.5,
    'RandomModel-0h-SampleSet-SENSITIVITY': 0.5,
    'RandomModel-0h-SampleSet-SPECIFICITY': 0.5,
    'RandomModel-0h-SampleSet-PRECISION': 0.5,
    'RandomModel-0h-SampleSet-AUC': 0.5,

    'RandomModel-3h-TestSet-SENSITIVITY':0.5,
    'RandomModel-3h-TestSet-SPECIFICITY': 0.5,
    'RandomModel-3h-TestSet-PRECISION': 0.5,
    'RandomModel-3h-TestSet-AUC': 0.5,
    'RandomModel-3h-SampleSet-SENSITIVITY': 0.5,
    'RandomModel-3h-SampleSet-SPECIFICITY': 0.5,
    'RandomModel-3h-SampleSet-PRECISION': 0.5,
    'RandomModel-3h-SampleSet-AUC': 0.5,

    'CoxPHM-0h-TestSet-SENSITIVITY': 0.8256,
    'CoxPHM-0h-TestSet-SPECIFICITY': 0.8256,
    'CoxPHM-0h-TestSet-PRECISION': 0.8833,
    'CoxPHM-0h-TestSet-AUC': 0.95,
    'CoxPHM-0h-SampleSet-SENSITIVITY': 0.815,
    'CoxPHM-0h-SampleSet-SPECIFICITY': 0.936,
    'CoxPHM-0h-SampleSet-PRECISION': 0.871,
    'CoxPHM-0h-SampleSet-AUC': 0.929,

    'CoxPHM-3h-TestSet-SENSITIVITY': 0.88,
    'CoxPHM-3h-TestSet-SPECIFICITY': 0.9867,
    'CoxPHM-3h-TestSet-PRECISION': 0.9333,
    'CoxPHM-3h-TestSet-AUC': 0.92,
    'CoxPHM-3h-SampleSet-SENSITIVITY': 0.856,
    'CoxPHM-3h-SampleSet-SPECIFICITY': 0.987,
    'CoxPHM-3h-SampleSet-PRECISION': 0.936,
    'CoxPHM-3h-SampleSet-AUC': 0.984,

    'High_LSTM-0h-TestSet-SENSITIVITY': 0.9625,
    'High_LSTM-0h-TestSet-SPECIFICITY': 0.8054,
    'High_LSTM-0h-TestSet-PRECISION': 0.8676,
    'High_LSTM-0h-TestSet-AUC': 0.9423,
    'High_LSTM-0h-SampleSet-SENSITIVITY': 0.996,
    'High_LSTM-0h-SampleSet-SPECIFICITY': 0.998,
    'High_LSTM-0h-SampleSet-PRECISION': 0.99709,
    'High_LSTM-0h-SampleSet-AUC': 0.99,

    'Medium_LSTM-0h-TestSet-SENSITIVITY': 0.925,
    'Medium_LSTM-0h-TestSet-SPECIFICITY': 0.7022,
    'Medium_LSTM-0h-TestSet-PRECISION': 0.7905,
    'Medium_LSTM-0h-TestSet-AUC': 0.86,
    'Medium_LSTM-0h-SampleSet-SENSITIVITY': 0.9416,
    'Medium_LSTM-0h-SampleSet-SPECIFICITY': 0.862,
    'Medium_LSTM-0h-SampleSet-PRECISION': 0.898,
    'Medium_LSTM-0h-SampleSet-AUC': 0.96782,

    'Low_LSTM-0h-TestSet-SENSITIVITY': 0.2546,
    'Low_LSTM-0h-TestSet-SPECIFICITY': 0.9275,
    'Low_LSTM-0h-TestSet-PRECISION': 0.661,
    'Low_LSTM-0h-TestSet-AUC': 0.7665,
    'Low_LSTM-0h-SampleSet-SENSITIVITY': 0.232,
    'Low_LSTM-0h-SampleSet-SPECIFICITY': 0.996,
    'Low_LSTM-0h-SampleSet-PRECISION': 0.649,
    'Low_LSTM-0h-SampleSet-AUC': 0.885,

    'High_LSTM-3h-TestSet-SENSITIVITY': 0.9835,
    'High_LSTM-3h-TestSet-SPECIFICITY': 0.7898,
    'High_LSTM-3h-TestSet-PRECISION': 0.894,
    'High_LSTM-3h-TestSet-AUC': 93.53,
    'High_LSTM-3h-SampleSet-SENSITIVITY': 0.9986,
    'High_LSTM-3h-SampleSet-SPECIFICITY': 1,
    'High_LSTM-3h-SampleSet-PRECISION': 0.999,
    'High_LSTM-3h-SampleSet-AUC': 1,

    'Medium_LSTM-3h-TestSet-SENSITIVITY': 0.9379,
    'Medium_LSTM-3h-TestSet-SPECIFICITY': 0.693,
    'Medium_LSTM-3h-TestSet-PRECISION': 0.824,
    'Medium_LSTM-3h-TestSet-AUC': 0.8395,
    'Medium_LSTM-3h-SampleSet-SENSITIVITY': 0.9726,
    'Medium_LSTM-3h-SampleSet-SPECIFICITY': 0.8646,
    'Medium_LSTM-3h-SampleSet-PRECISION': 0.9186,
    'Medium_LSTM-3h-SampleSet-AUC': 0.9478,

    'Low_LSTM-3h-TestSet-SENSITIVITY': 0.9793,
    'Low_LSTM-3h-TestSet-SPECIFICITY': 0.3259,
    'Low_LSTM-3h-TestSet-PRECISION': 0.677,
    'Low_LSTM-3h-TestSet-AUC': 0.7337,
    'Low_LSTM-3h-SampleSet-SENSITIVITY': 0.992,
    'Low_LSTM-3h-SampleSet-SPECIFICITY': 0.39266,
    'Low_LSTM-3h-SampleSet-PRECISION': 0.692,
    'Low_LSTM-3h-SampleSet-AUC': 0.77055
}

if __name__ == '__main__':

    df = pd.DataFrame(columns=columns, dtype='object')
    model_id = 1
    for model_name in ['RandomModel', 'CoxPHM','High_LSTM', 'Medium_LSTM', 'Low_LSTM']:
        for time_period in ['0h','3h']:
            for dataset in ['TestSet','SampleSet']:
                row_data = []
                row_data.append(model_id)
                row_data.append(model_name)
                row_data.append(time_period)
                row_data.append(dataset)
                sen = data_dict.get(f'{model_name}-{time_period}-{dataset}-SENSITIVITY')
                spe = data_dict.get(f'{model_name}-{time_period}-{dataset}-SPECIFICITY')
                pre = data_dict.get(f'{model_name}-{time_period}-{dataset}-PRECISION')
                auc = data_dict.get(f'{model_name}-{time_period}-{dataset}-AUC')
                row_data.append(sen)
                row_data.append(spe)
                row_data.append(pre)
                row_data.append(auc)
                df.loc[len(df)] = row_data
        model_id +=1
    df.to_csv(root + '\\datasets\\Model_Property.csv', encoding='gbk', index=False)
