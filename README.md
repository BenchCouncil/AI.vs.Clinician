# AI-Assisted Clinician vs. Clinician: Unveiling the Intricate Interactions Between AI and Clinicians Through a Prospective, Multi-Site, Mixed-Blind, Multi-Arm, Randomized, and Controlled Trial for Early Sepsis Warning.

Since December 2022, we have been conducting an AI-based clinical trial on the diagnosis and treatment of sepsis across 14 medical centers, with a trial period of 18 months. The study involves 7,500 diagnostic procedures performed by 125 clinicians. To ensure transparency in the trial, we have recorded all actions of the clinicians and compiled them into the dataset "AI vs. Clinician". "AI vs. Clinician" is a large and human-centered database that comprises information related to the behavior variations of clinicians’ diagnosis with or without the assistance of different AI models. Furthermore, we conducted an in-depth analysis of the trial results to reveal the real-world situation of AI-based medical devices in clinical practice, as presented in our paper “Assisting Clinicians With AI Systems for Early Sepsis Warning: A Prospective, Multi-Site, Mixed-Blind, Multi-Arm, Randomized, Multi-Controlled Trial”.

# Contents
- [Data Prerequisites](#data-prerequisites)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)


# Data Prerequisites

Our work involves publicly available data provided by the MIMIC database; therefore, users must first obtain the appropriate access and usage permissions from the relevant parties of the MIMIC database.

**1. PhysioNet Account [Link](https://physionet.org/)**

First, apply for an account on the PhysioNet platform. Then, proceed to take the CITI PROGRAM exam and obtain the exam report. With the exam report in hand, you can apply for database usage permission on the MIMIC official website.

After completing the aforementioned steps, you can download the relevant MIMIC datasets.

**Notice:**  To access AI.vs.Clinician database [AI.vs.Clinician Database](https://www.benchcouncil.org/ai.vs.clinician/), one more step is needed: send an access request to the contributors and provide a description of the research project.

**2.MIMIC-III 1.4 Database [Link](https://physionet.org/content/mimiciii/1.4/)**

MIMIC-III is a large-scale clinical intensive care database that is freely available and openly shared. The data covers the period from 2001 to 2012.

**3.MIMIC-IV 2.2 Database [Link](https://physionet.org/content/mimiciv/2.2/)**

MIMIC-IV is an updated version of MIMIC-III, with data covering the period from 2008 to 2019.


**4.MIMIC-CXR-JPG 2.0.0 Database [Link](https://physionet.org/content/mimic-cxr/2.0.0/)**

The MIMIC-CXR-JPG database is a collection of chest X-ray images in JPG format, corresponding to patients in the MIMIC-IV dataset.


**5.MIMIC-IV-NOTE 2.2 Database [Link](https://physionet.org/content/mimic-iv-note/2.2/)**

The MIMIC-IV-Note primarily consists of discharge summaries and imaging text reports for patients in the MIMIC-IV dataset.




# Repo Contents

**1. mimiciv-sample**  The code repository contains patient data selected from MIMIC-IV for clinical trials.
 
**2. mimic-lstm** This directory contains all the code for the development of AI models based on LSTM.

**3. mimic-coxphm** This directory contains all the code for the development of AI models based on the traditional COX model.

**4. ai-vs-clinician**  This directory contains the organization of the procedural data from the trial.

**5. analysis**  This directory contains part of the analysis data and code of the paper "Assisting Clinicians With AI Systems for Early Sepsis Warning: A Prospective, Multi-Site, Mixed-Blind, Multi-Arm, Randomized, Multi-Controlled Trial".



# System Requirements

**Hardware Requirements**

The package requires a server for AI model training and a standard computer for analysis. The package has been tested on the following systems:

**（1）Model Training Environment**
- **RAM:** 62 GB
- **CPU:** 24 cores
- **GPU:** Tesla V100-PCIE-32G
- **Operating System:** Linux Ubutu 16.04.7 LTS

**（2）Analysis Environment**

      ① Linux Environment
      - **RAM:** 62 GB
      - **CPU:** 24 cores
      - **Operating System:** Linux Ubutu 16.04.7 LTS

      AND
      
      ② Windows Environment
      - **RAM:** 16.0 GB
      - **CPU:** Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz   2.30 GHz
      - **Operating System:** Windows 10 Home Chinese Version


**Software Requirements**

 Python 3.8
 
 Pycharm 2020


# Installation Guide

Change the directory, install the dependencies for each part, and refer to the README of each part for the process:

### (1) Construct patient cohort from MIMIC-IV for clinical trials:

`cd ./mimiciv-sample`

`pip install -r requirements.txt`

This section refers to the README, which can be accessed via the link. [Link](https://github.com/BenchCouncil/AI.vs.Clinician/blob/master/mimiciv-sample/README.md)

### (2) AI model training

**LSTM**

`cd ./mimic-lstm`

`pip install -r requirements.txt`

This section refers to the README, which can be accessed via the link. [Link](https://github.com/BenchCouncil/AI.vs.Clinician/blob/master/mimic-lstm/README.md)

**CoxPHM**

`cd ./mimic-coxphm`

`pip install -r requirements.txt`

This section refers to the README, which can be accessed via the link. [Link](https://github.com/BenchCouncil/AI.vs.Clinician/blob/master/mimic-coxphm/README.md)

The total time for installation of requirements.txt is within 2 hours.

The patient cohort construction and AI model training would require over 12 hours.

### (3) Organization of the procedural data from the trial

`cd ./ai-vs-clinician`

This section refers to the README, which can be accessed via the link. [Link](https://github.com/BenchCouncil/AI.vs.Clinician/edit/master/ai-vs-clinician/README.md)

### (4) Analysis 

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


# Demo

(1) Changes the current directory to `analysis`.
   
   `cd analysis/`
   
 (2) Runs the Python script.
   
    `python main_paper2_no3hback.py parameter1_project_path parameter2_analysis_operation`
 
    The command to run is as follows:
 
       (1) Analyze the AUC and CI of the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "auc"`
       
       (2) Analyze the sensitivity and CI of the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "sensitivity"`
      
       (3) Analyze the specificity and CI of the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "specificity"`
      
       (4) Analyze the accuracy and CI of the clinician's diagnosis in relation to antibiotic use.
       `python main_paper2_no3hback.py /path/to/project "antacc"`
      
       (5) Analyze the time and CI of the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "diagtime"`
      
       (6) Analyze the number of revisions and CI to the clinician's diagnosis.
       `python main_paper2_no3hback.py /path/to/project "diag_modify_times"`
       
       (7) Analyze the number of historical examinations and CI reviewed by the clinician.
       `python main_paper2_no3hback.py /path/to/project "check_hisnum"`
       
       (8) Analyze the number of current examinations and CI reviewed by the clinician.
       `python main_paper2_no3hback.py /path/to/project "check_nextnum"`

       
       
 (3) Explanation of Output Values.

 The output values represent statistical results regarding different AI models, different blinding methods, and classifications of different clinicians (hospital level, gender, age, years of practice, and position).
 ![Demo output example]([image_path_or_url](https://github.com/BenchCouncil/AI.vs.Clinician/blob/master/demo-output-example.jpg))

 
 The demo running would require about 2 minutes.
































