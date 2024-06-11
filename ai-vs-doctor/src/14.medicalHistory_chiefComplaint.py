import pandas as pd
from src.constant import *
import ast


columns = ['PATIENT_CASE_ID',  'ITEM',  'VALUE']


if __name__ == '__main__':
    df_sample_all = pd.read_csv(fn_sample_all, encoding='gbk')
    df_sample = df_sample_all.drop_duplicates(subset=['START_ENDTIME'])

    df_case_info = pd.read_csv(root + 'datasets\\Patient_Case_Info.csv', encoding='gbk')
    df = pd.DataFrame(columns=columns, dtype='object')

    for index, row in df_sample.iterrows():
        print(index)
        df_case_info_id = df_case_info[df_case_info['CURRENT_TIME'] == row['START_ENDTIME']]
        case_id = df_case_info_id.iloc[0]['PATIENT_CASE_ID']

        for item,value in [('过去病史',row['病史']),('主诉',row['主诉']),('现病史',row['现病史'])]:
            if str(value) != 'nan':
                row_data = []
                row_data.append(case_id)
                row_data.append(item)
                if item == '现病史':
                    ind = value.find('现病史（原英文）:')
                    value_after = value[ind+9:]
                elif item == '主诉':
                    ind = value.find('英文：')
                    value_after = value[ind+3:]
                else:
                    value_after = value
                row_data.append(value_after)
                df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\MedicalHistory_ChiefComplaint.csv', encoding='gbk', index=False)


