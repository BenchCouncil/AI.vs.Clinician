#!/bin/bash

python 1-sample_selection_preprocess.py &
wait

python 2-sample_selection_21_ill.py &
wait

python 2-sample_selection_22_not.py &
wait

python 3-mimiciv_ill_not_radio.py &
wait

python 4-sample_selection_final_chartname_check.py &
wait

python 5-select_3000_from_dist_final.py&
wait

python 6-mimiciii_sample_remove.py
wait

python 7-mimiciv_sample_fill_data_ill.py
wait

python 7-mimiciv_sample_fill_data_not.py
wait

python 8-mimiciv_sample_fill_doctor_data_final.py
wait

python 9-mimiciv_train.py
wait

python 10-mimiciv_sample_add_ethnicity.py
wait

python 11-mimiciv_mechvent.py
wait

python 12-sample_selection_doctor_data_final_chartname.py
wait

python 13-change_units.py

