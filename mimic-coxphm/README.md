| No. | Code | Description |
| ------- | ------- | ------- |
| |requirements.txt|Install environment.|
| |cd mimic-coxphm/1-data_process/ |Switch Path.|
|1|1_tscore_mimiciii_model_data_ill.py|CoxPHM Input data for septic patients in MIMIC-III.|
|2|1_tscore_mimiciii_model_data_not.py|CoxPHM Input data for non-septic patients in MIMIC-III.|
|3|2_tscore_mimiciv_model_data_ill.py|CoxPHM Input data for septic patients in MIMIC-IV.|
|4|2_tscore_mimiciv_model_data_not.py|CoxPHM Input data for non-septic patients in MIMIC-IV.|
|5|3-mimiciv_sample_diff_meanvalue.py|CoxPHM Input data for sample in MIMIC-IV.|
|6|4-tscore_connect_mimiciii_iv.py|Merge the input data from MIMIC-III and MIMIC-IV.|
| |cd mimic-coxphm/2-cph-newfill-final/ |Switch Path.|
|7|cox-train.py|Model training.|
|8|cox-infer-test.py|Model inference on the test set.|
|9|cox-infer-4800sample-no12h.py|Model inference on the sample.|