#!/bin/bash

python3.7 1_tscore_mimiciii_model_data_ill.py
python3.7 1_tscore_mimiciii_model_data_not.py

python3.7 2_tscore_mimiciv_model_data_ill.py
python3.7 2_tscore_mimiciv_model_data_not.py

python3.7 4-tscore_connect_mimiciii_iv.py

python3.7 3-mimiciv_sample_diff_meanvalue.py

