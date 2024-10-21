# AI-Assisted Clinician vs. Clinician: Unveiling the Intricate Interactions Between AI and Clinicians Through a Prospective, Multi-Site, Mixed-Blind, Multi-Arm, Randomized, and Controlled Trial for Early Sepsis Warning.

AI.vs.Clinician is a large and human-centered database that comprises information related to the behavior variations of clinicians’ diagnosis with or without the assistance of different AI models.

The database is publicly available from PhysioNet (awaiting approval). This repo contains all the related source code used during the construction of this database.

The details and steps to use AI.vs.Clinician database are as follows.

# Request to Access MIMIC databases and AI.vs.Clinician database

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


# 1.mimiciv-sample
The code repository consists of sepsis patients selected from MIMIC-IV.
See the readme in the section for detailed steps.

# 2.mimic-lstm
The LSTM predictions for the selected sepsis patients mentioned above.
See the readme in the section for detailed steps.

# 3.mimic-coxphm
The CoxPHM predictions for the selected sepsis patients mentioned above.
See the readme in the section for detailed steps.

# 4.ai-vs-clinician
AI.vs.Clinician, A Freely Accessible Human-AI Interaction Database for AI in Medicine.
See the readme in the section for detailed steps.
