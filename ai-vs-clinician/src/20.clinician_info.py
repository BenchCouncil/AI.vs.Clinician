import pandas as pd
from src.constant import *

columns = ['Clinician_ID', 'INSTITUTION_LEVEL', 'GENDER', 'AGE', 'DEPARTMENT', 'YEARS_WORKED', 'CLASS_OF_POSITION',  'AREA_OF_EXPERTISE']



if __name__ == '__main__':


    df_doctor = pd.read_csv(root + 'data\\doctor\\doctor_info.csv',encoding='gbk')

    df = pd.DataFrame(columns=columns, dtype='object')
    for index,row in df_doctor.iterrows():
        log_id = row['doctor_logid']
        unit = row['institution_level']
        gender = row['gender']
        age = row['age']
        year = row['years_worked']
        depart = row['department']
        position = row['class_of_position']
        area = row['area_of_expertise']

        row_data = []
        row_data.append(log_id)
        row_data.append(unit)
        row_data.append(gender)
        row_data.append(age)
        row_data.append(depart)
        row_data.append(year)
        row_data.append(position)
        row_data.append(area)
        df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Clinician_Info.csv', encoding='gbk', index=False)
