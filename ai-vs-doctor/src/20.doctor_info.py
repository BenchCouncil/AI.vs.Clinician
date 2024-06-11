import pandas as pd
from src.constant import *

columns = ['DOCTOR_ID', 'INSTITUTION_LEVEL', 'GENDER', 'AGE', 'DEPARTMENT', 'YEARS_WORKED', 'CLASS_OF_POSITION',  'AREA_OF_EXPERTISE']

doctor_unit_dict = {
    # 二甲医院
    "全州县人民医院": 1,
    "兴安县人民医院": 1,
    "蒙山县人民医院": 1,
    "全州县中医医院": 1,
    "河池市第三人民医院": 1,
    "恭城瑶族自治县人民医院": 1,
    "广西桂林兴安县人民医院": 1,
    "灵川县人民医院": 1,
    "永福县人民医院": 1,
    "广西省梧州市蒙山县人民医院": 1,
    "广西兴安县人民医院": 1,
    "广西桂林市兴安县人民医院":1,
    "桂林市兴安县人民医院": 1,
    "桂林兴安县人民医院": 1,
    "桂林市全州县人民医院": 1,
    "隆林各族自治县中医医院": 1,
    "广西壮族自治区兴安县人民医院": 1,
    "灌阳县人民医院": 1,
    # 三甲医院
    "桂林医学院附属医院": 2,
    "桂林市人民医院": 2,
    "桂林医学院第二附属医院": 2,
    "桂林医学院第二人附属医院": 2,
    # 医学院
    "桂林医学院": 3,
}

if __name__ == '__main__':


    df_doctor = pd.read_csv(root + 'data\\doctor\\doctor_info.csv')

    df = pd.DataFrame(columns=columns, dtype='object')
    for index,row in df_doctor.iterrows():
        log_id = row['系统日志中医生ID']
        unit = row['工作单位']
        gender = row['性别']
        age = row['年龄']
        year = row['从业年限']
        depart = row['科室']
        position = row['职称']
        area = row['擅长专业']

        row_data = []
        row_data.append(log_id)
        row_data.append(doctor_unit_dict.get(unit))
        row_data.append(gender)
        row_data.append(age)
        row_data.append(depart)
        row_data.append(year)
        row_data.append(position)
        row_data.append(area)
        df.loc[len(df)] = row_data

    df.to_csv(root + '\\datasets\\Doctor_Info.csv', encoding='gbk', index=False)
