#
cd 1-data_process/
./run.sh

#train model
python cox-train.py

#test predict
python cox-infer-test.py

#sample predict
python cox-infer-4800sample-no12h.py