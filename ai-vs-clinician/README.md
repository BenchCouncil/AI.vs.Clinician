**`Objective:`**Restructuring the database into 22 tables. (14 tables for patient information, 5 tables for AI model information, 3 tables for clinician information)

# Patient Information Tables
| No. | Table Name | Description |
| ------- | ------- | ------- |
|1 |Patient_Case_Info|The information for each patient case, including the identifier information, demographics,admission time, sepsis or non-sepsis, sepsis onset time, and current time window.|
|2 |Patient_Fundamental_ITEM|The current and history information of seven fundamental examination items for patient cases.|
|3 |FLUID_INPUT|The latest 24-hour and history fluid input information for each patient case.|
|4 |FLUID_OUTPUT|The latest 24-hour and history fluid output information for each patient case.|
|5 |Complete_Blood_Count|The current and history complete blood count information for each patient case.|
|6 |Pathogen_Blood|The current and history pathogen blood information for each patient case.|
|7 |Medical_Imaging|The current and history medical imaging information for each patient case.(The IMAGE_TYPE column in the table is for reference only and is not completely accurate.)|
|8 |Arterial_Blood_Gas_Analysis|The current and history arterial blood gas analysis data for each patient case.|
|9 |Haemostatic_Function|The current and history haemostatic data for each patient case.|
|10|Culture_Smear|The current and history culture smear information for each patient case.|
|11|Procalcitonin|The current and history procalcitonin information for each patient case.|
|12|SOFA|The current SOFA score and specific variables for each patient case.|
|13|Drug_History|The drug history information for each patient case.|
|14|MedicalHistory_ChiefComplaint|The medical history and chief complaint information for each patient case.|


# AI Model Information Tables
| No. | Table Name | Description |
| ------- | ------- | ------- |
|1|Model_Property|The properties of each AI model, including the model id, model name, the sensitivity, specificity, precision, and AUC on training, validating, and testing dataset.|
|2|Model_Dataset|The patient IDs (HADM_ID) used as the training set, validating set, or testing set for all the AI models. |
|3|CoxPHM_Feature|The feature input for each patient case to train a CoxPHM model.|
|4|LSTM_Feature|The feature input for each patient case to train an LSTM model.|
|5|Model_Cohort_Infer|The AI models’ inference results on the patient cohort, including the decision, the probability of sepsis onset presently (0h), and the probability of sepsis onset within the next three hours (3h).|


# Clinician Information Tables
| No. | Table Name | Description |
| ------- | ------- | ------- |
|1|Clinician_Info|The information for each clinician, including the identifier information, demographics, department, years of working, class of position, and area of expertise.|
|2|Clinician_Click_Behavior|The information of clinician’s click behaviors, including the clicked examination item for viewing, and the clicked time. |
|3|Clinician_Diagnosis|Each clinician’s preliminary and final diagnosis decisions on each patient case with or without the aid of an AI model, including the clicked sequence (link with Clinician_Click_Behavior table), preliminary decision, final decision, and corresponding timestamps.|