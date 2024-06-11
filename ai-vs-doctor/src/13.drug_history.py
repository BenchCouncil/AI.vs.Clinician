import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID', 'STARTTIME', 'DRUG']


if __name__ == '__main__':
    df_sample_all = pd.read_csv(fn_sample_all, encoding='gbk')
    df_sample = df_sample_all.drop_duplicates(subset=['START_ENDTIME'])

    df_case_info = pd.read_csv(root + 'datasets\\Patient_Case_Info.csv', encoding='gbk')
    df = pd.DataFrame(columns=columns, dtype='object')

    for index, row in df_sample.iterrows():
        print(index)
        df_case_info_id = df_case_info[df_case_info['CURRENT_TIME'] == row['START_ENDTIME']]
        case_id = df_case_info_id.iloc[0]['PATIENT_CASE_ID']

        drug_list = row['历史用药']
        if str(drug_list) != 'nan':
            drug_list = eval(drug_list)
            for drug in drug_list:
                row_data = []
                row_data.append(case_id)
                row_data.append(drug.get('开始时间'))

                value = drug.get('药名')
                ind = value.find('|')
                value_after = value[ind+1:]
                row_data.append(value_after)
                df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Drug_History.csv', encoding='gbk', index=False)


