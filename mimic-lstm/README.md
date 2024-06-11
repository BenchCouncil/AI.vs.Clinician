| No. | Code | Description |
| ------- | ------- | ------- |
| |requirements.txt|Install environment.|
| |cd mimic-lstm/1_preprocess/ |Switch Path.|
|1|1_lstm_mimiciii_model_data_preprocess.py|Preprocessing of MIMIC-III.|
|2|2_lstm_model_data_fill_meanvalue_ill.py|Mapping table for supplemental data of septic patients.|
|3|2_lstm_model_data_fill_meanvalue_not_death.py|Mapping table for supplemental data of deceased non-septic patients.|
|4|2_lstm_model_data_fill_meanvalue_not_nodeath.py|Mapping table for supplemental data of non-septic patients who are alive.|
|5|3_lstm_mimiciii_model_data_ill.py|Generate hourly data for septic patients in MIMIC-III.|
|6|3_lstm_mimiciii_model_data_not.py|Generate hourly data for non-septic patients in MIMIC-III.|
|7|4_lstm_mimiciii_model_data_process_0h.py|LSTM Input data for MIMIC-III at onset (0h) for the illness model.|
|8|4_lstm_mimiciii_model_data_process_3h.py|LSTM Input data for MIMIC-III at 3 hours after onset for the illness model.|
|9|5-lstm_mimiciv_model_data_preprocess.py|Preprocessing of MIMIC-IV.|
|10|6_lstm_mimiciv_model_data_sample_ill_not.py|Generate hourly results for septic and non-septic patients in the MIMIC-IV sample.|
|11|6_lstm_mimiciv_model_data_train_ill_not.py|Generate hourly results for septic and non-septic patients in the MIMIC-IV training set.|
|12|7_lstm_mimiciv_model_data_train_0h.py|LSTM Input data for MIMIC-IV training set at onset (0h) for the illness model.|
|13|7_lstm_mimiciv_model_data_train_3h.py|LSTM Input data for MIMIC-IV training set at 3 hours after onset for the illness model.|
|14|8_lstm_mimiciv_model_data_sample_0h.py|LSTM Input data for MIMIC-IV sample at onset (0h) for the illness model.|
|15|8_lstm_mimiciv_model_data_sample_3h.py|LSTM Input data for MIMIC-IV sample at 3 hours after onset for the illness model.|
|16|8_lstm_mimiciv_model_data_sample_add_hadmid.py|Sample prediction input data with HADMID and time period added.|
| |cd mimic-lstm/ |Switch Path.|
|17|4_rnn_mimic_sepsis.py|Model training.|
|18|4_rnn_mimic_sepsis_final_predict.py|Model prediction.|