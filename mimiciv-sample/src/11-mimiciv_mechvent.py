import pandas as pd
import os
# 在mimic iv中找机械通气的患者


PATH = '/home/ddcui/'
ROOT = f'{PATH}mimic-original-data/mimiciv_database/'
TO_ROOT = f'{PATH}/hai-med-database/mimiciv-sample/data/'


def set_mimiciv_mechvent():
    print("reading df_chartevents")
    item_id = [223848, 223849,225303,225792,225794,226260,227061,227565,227566,229314]

    for i, df_chunk in enumerate(
            pd.read_csv(ROOT + 'chartevents.csv', usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'value'],iterator=True, chunksize=10000000)):
        print(i)
        df_chunk = df_chunk[df_chunk['itemid'].isin(item_id)]

        filename = TO_ROOT+'preprocess/mimiciv_all_mech_vent_subject.csv'
        if not os.path.exists(filename):
            df_chunk.to_csv(filename, mode='w', index=False)
        else:
            df_chunk.to_csv(filename, mode='a', header=False, index=False)

if __name__ == '__main__':
    set_mimiciv_mechvent()