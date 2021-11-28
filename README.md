# Kolluri-Vechicle_Insurance_Classifier

# Dataset: 

Information about Dataset:
=========================

Demographics (gender, age, region code type),

Vehicles (Vehicle Age, Damage),

Policy (Premium, sourcing channel) etc.


Attributes
===========

id -> unique identity (int)             

Gender ->  Male/Female (object-string) 

Age ->  Age (int)

Driving_License -> 0/1 (binary)         

Region_Code -> (int)

Previously_Insured -> 0/1 (binary)     

Vehicle_Age -> (object)

Vehicle_Damage -> Yes/No (binary) 

Annual_Premium -> (int)

Policy_Sales_Channel  -> (int) 

Vintage -> (int)

Response -> 0/1 (binary) (response)        


Task
====
**Predict whether customer is interested in vehicle insurance based on previous vehicle insurance.**

Oversampling and Undersampling 
==============================
* Smote (Synthetic Minority Oversampling Technique)-(OverSampling).
* NearMiss (UnderSampling).

Performed Models
================
* Model-1 : Logistic Regression.
* Model-2 : RandomForest.
* Model-3 : Gradient Boosting.
* Hyperparameter Tuning 

