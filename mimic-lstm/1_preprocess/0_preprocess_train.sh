#!/bin/bash

#python3.7 2_lstm_model_data_fill_meanvalue_ill.py
#python3.7 2_lstm_model_data_fill_meanvalue_not_death.py
#python3.7 2_lstm_model_data_fill_meanvalue_not_nodeath.py

#python3.7 3_lstm_mimiciii_model_data_ill.py
#python3.7 3_lstm_mimiciii_model_data_not.py

folder_name="diff_meanvalue_del_missdata_0.4"

if [ ! -d "$folder_name" ]; then
    mkdir "$folder_name"

python3.7 4_lstm_mimiciii_model_data_process_0h.py
python3.7 4_lstm_mimiciii_model_data_process_3h.py

#python3.7 6_lstm_mimiciv_model_data_train_ill_not.py
#python3.7 6_lstm_mimiciv_model_data_sample_ill_not.py

python3.7 7_lstm_mimiciv_model_data_train_0h.py
python3.7 7_lstm_mimiciv_model_data_train_3h.py

python3.7 8_lstm_mimiciv_model_data_sample_0h.py
python3.7 8_lstm_mimiciv_model_data_sample_3h.py

python3.7 8_lstm_mimiciv_model_data_sample_add_hadmid.py
