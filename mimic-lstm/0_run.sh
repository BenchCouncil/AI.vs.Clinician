#
cd 1_preprocess/
./0_preprocess_train.sh

#train model
python 4_rnn_mimic_sepsis.py

#model predict
python 4_rnn_mimic_sepsis_final_predict.py
