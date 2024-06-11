# 1.Sepsis Oneset time
The onset time of sepsis was based on the Sepsis 3.0 rules for the diagnosis of sepsis.

**1.1.MIMIC-III sepsis oneset time**

According to the 2018 paper published in Nature:
Paper：Komorowski M, Celi L A, Badawi O, et al. The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care[J]. Nature medicine, 2018, 24(11): 1716-1720.
[Paper](https://www.nature.com/articles/s41591-018-0213-5)

[Github(Matlab) Link](https://github.com/matthieukomorowski/AI_Clinician)

[Github(Python) Link](https://github.com/uribyul/py_ai_clinician)

**The steps for reproduction are as follows:**

(1) Step Zero: Prepare the Postgres database following the instructions in the readme tutorial.

(2) Step One (Data Extraction): Extract the data, resulting in 24 CSV files.

(3) Step Two (Define Sepsis Cohort): Define the sepsis cohort, resulting in sepsis_mimiciii.csv.

(4) Convert the onset timestamps in sepsis_mimiciii.csv to date format according to UTC time.

This file contains the sepsis cohort required for this study from MIMIC-III, including the onset time of illness.


**1.2.MIMIC-IV sepsis oneset time**

The timing of the onset of sepsis in MIMIC-IV is a continuation of the above work, Github open source code.
[Github(Python) Link](https://github.com/cmudig/AI-Clinician-MIMICIV)

**The steps for reproduction are as follows:**

(1) Obtain the MIMIC-IV database from Google Cloud following the instructions provided in the readme tutorial in the source code.

(2) Locate the executable file run.sh in the top-level directory of the project.

(3) Follow the steps outlined in the run.sh file, including executing echo "7/11 BUILD SEPSIS COHORT", to obtain sepsis_cohort.csv.

(4) sepsis_cohort.csv contains the onset time of sepsis patients, serving as the sepsis cohort required for this study from MIMIC-IV.

(5) Convert the onset timestamps in sepsis_cohort.csv to date format according to UTC time.
This file represents the sepsis cohort required for this study from MIMIC-IV, containing the onset time of illness. 
Notice: The step echo "2/11 CALCULATE SEPSIS ONSET" in run.sh does not provide the final onset time of sepsis patients. The resulting sepsis_onset.csv from this step represents suspected onset times of sepsis. Proceed with the subsequent steps without confusion.


# 2.The process of getting Sample is as follows：
| Sequence | Code | Description | Note|
| ------- | ------- | ------- | ------- |
|1|1-sample_selection_preprocess.py | Pre-processing, generation of adverse event forms, etc. |
|2|2-sample_selection_21_ill.py| Filtering for data-rich septic patients.|
|3|2-sample_selection_22_not.py|Filtering for data-rich non-septic patients.|
|4|3-mimiciv_ill_not_radio.py|Filtering for patients with existing chest X-rays (CXR).|
|5|4-sample_selection_final_chartname_check.py| Reconfirming data richness.|
|6|5-select_3000_from_dist_final.py|Commencing sample selection.|
|7|6-mimiciii_sample_remove.py|Remove duplicates between the samples from MIMIC-IV and the data from MIMIC-III.|
|8|7-mimiciv_sample_fill_data_ill.py|Supplementary information for septic patients' data of interest to doctors. |
|9|7-mimiciv_sample_fill_data_not.py|Supplementary information for non-septic patients' data of interest to doctors.|
|10|8-mimiciv_sample_fill_doctor_data_final.py|Generate patient data for the sample.|
|11|9-mimiciv_train.py|Remove training data from the MIMIC-IV sample.|
|12|10-mimiciv_sample_add_ethnicity.py|Add demographic information for the sample.|
|13|11-mimiciv_mechvent.py|Generate patients from MIMIC-IV who underwent mechanical ventilation.|
|14|12-sample_selection_doctor_data_final_chartname.py|Translate and standardize the names of inspection items into Chinese, then generate patient data.|
|15|13-change_units.py|Standardize unit conversions.|
