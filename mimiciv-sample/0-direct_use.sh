#!/bin/bash

#Directly generate the data of the selected sample
python 12-sample_selection_doctor_data_final_chartname.py &
wait

#If you need to modify the unit of the data, please refer to the following function
#python 13-change_units.py &
#wait
