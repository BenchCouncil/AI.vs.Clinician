# AI-Assisted Clinician vs. Clinician: Unveiling the Intricate Interactions Between AI and Clinicians Through a Prospective, Multi-Site, Mixed-Blind, Multi-Arm, Randomized, and Controlled Trial for Early Sepsis Warning.

Since December 2022, we have been conducting an AI-based clinical trial on the diagnosis and treatment of sepsis across 14 medical centers, with a trial period of 18 months. The study involves 7,500 diagnostic procedures performed by 125 clinicians. To ensure transparency in the trial, we have recorded all actions of the clinicians and compiled them into the dataset "AI vs. Clinician". "AI vs. Clinician" is a large and human-centered database that comprises information related to the behavior variations of clinicians’ diagnosis with or without the assistance of different AI models. Furthermore, we conducted an in-depth analysis of the trial results to reveal the real-world situation of AI-based medical devices in clinical practice, as presented in our paper “Assisting Clinicians With AI Systems for Early Sepsis Warning: A Prospective, Multi-Site, Mixed-Blind, Multi-Arm, Randomized, Multi-Controlled Trial”.


# Prerequisite Data Requirements

Our work involves publicly available data provided by the MIMIC database; therefore, users must first obtain the appropriate access and usage permissions from the relevant parties of the MIMIC database.

**1. PhysioNet Account**

First, apply for an account on the PhysioNet platform. Then, proceed to take the CITI PROGRAM exam and obtain the exam report. With the exam report in hand, you can apply for database usage permission on the MIMIC official website.

After completing the aforementioned steps, you can download the relevant MIMIC datasets.

**Notice:**  To access AI.vs.Clinician database, one more step is needed: send an access request to the contributors and provide a description of the research project.

**2.MIMIC-III 1.4 Database**

MIMIC-III is a large-scale clinical intensive care database that is freely available and openly shared. The data covers the period from 2001 to 2012.

**3.MIMIC-IV 2.2 Database**

MIMIC-IV is an updated version of MIMIC-III, with data covering the period from 2008 to 2019.


**4.MIMIC-CXR-JPG 2.0.0 Database**

The MIMIC-CXR-JPG database is a collection of chest X-ray images in JPG format, corresponding to patients in the MIMIC-IV dataset.


**5.MIMIC-IV-NOTE 2.2 Database**

The MIMIC-IV-Note primarily consists of discharge summaries and imaging text reports for patients in the MIMIC-IV dataset.




# Repo Contents

**1. mimic-lstm** This directory contains all the code for the development of AI models based on LSTM.

**2. mimic-coxphm** This directory contains all the code for the development of AI models based on the traditional COX model.

**3. mimiciv-sample**  The code repository contains patient data selected from MIMIC-IV for clinical trials.

**4. ai-vs-clinician**  This directory contains the organization of the procedural data from the trial.

**5. analysis**  This directory contains part of the analysis data and code of the paper "Assisting Clinicians With AI Systems for Early Sepsis Warning: A Prospective, Multi-Site, Mixed-Blind, Multi-Arm, Randomized, Multi-Controlled Trial".



# System Requirements

**Hardware Requirements**

**（1）Model Training Environment**
- **RAM:** 62 GB
- **CPU:** 24 cores
- **GPU:** Tesla V100-PCIE-32G
- **Operating System:** Linux

**（2）Analysis Environment**

      ① Linux Environment
      - **RAM:** 62 GB
      - **CPU:** 24 cores
        
      OR
      
      ② Windows Environment
      - **RAM:** 16.0 GB
      - **Operating System:** Windows 10


**Software Requirements**
- **Python Version:** Python 3.8
- **Analysis Tools:** Pycharm 2020


# Installation Guide
## Analysis Version
Navigate to the Analysis Directory：
`cd analysis/`

Install Required Packages：
`pip install -r requirements.txt`

Run：
`python main_paper2_no3hback.py parameter1_project_path parameter2_analysis_operation`
- **`parameter1_project_path`**
- **`parameter2_analysis_operation`**
  - `"auc"`
  - `"sensitivity"`
  - `"specificity"`
  - `"antacc"`
  - `"diagtime"`
  - `"diag_modify_times"`
  - `"check_hisnum"`
  - `"check_nextnum"`

which should successfully run within 5 seconds.

## Development Version
Change the directory, install the dependencies for each part, and refer to the README of each part for the process:

(1) `cd ./mimiciv-sample`
`pip install -r requirements.txt`
This section refers to the README, which can be accessed via the link. [链接](https://github.com/BenchCouncil/AI.vs.Clinician/blob/master/mimiciv-sample/README.md)

(2) `cd ./mimic-coxphm`
`pip install -r requirements.txt`
This section refers to the README, which can be accessed via the link.[链接](https://github.com/BenchCouncil/AI.vs.Clinician/blob/master/mimic-coxphm/README.md)

(3) `cd ./mimic-lstm`
`pip install -r requirements.txt`
This section refers to the README, which can be accessed via the link. [链接](https://github.com/BenchCouncil/AI.vs.Clinician/blob/master/mimic-lstm/README.md)

which should successfully run for more than 12 hours. 

# Demo

(1) Changes the current directory to `analysis`.
   
   `cd analysis/`
   
 (2) Runs the Python script.
   
    `python main_paper2_no3hback.py parameter1_project_path parameter2_analysis_operation`
 
    The command to run is as follows:
 
       (1) Analyze the AUC of the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "auc"`
       
       (2) Analyze the sensitivity of the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "sensitivity"`
      
       (3) Analyze the specificity of the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "specificity"`
      
       (4) Analyze the accuracy of the clinician's diagnosis in relation to antibiotic use.
       `python main_paper2_no3hback.py /path/to/project "antacc"`
      
       (5) Analyze the time of the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "diagtime"`
      
       (6) Analyze the number of revisions to the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "diag_modify_times"`
       
       (7) Analyze the number of historical examinations reviewed by the clinician.
       `python main_paper2_no3hback.py /path/to/project "check_hisnum"`
       
       (8) Analyze the number of current examinations reviewed by the clinician.
       `python main_paper2_no3hback.py /path/to/project "check_nextnum"`

       
       
 (3) Explanation of Output Values.

 The output values represent statistical results regarding different AI models, different blinding methods, and classifications of different clinicians (hospital level, gender, age, years of practice, and position).
 
































