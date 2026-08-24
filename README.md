Introduction: 
The problem under investigation for this project is to predict if a bank transaction is fraudulent. The dataset that was used to make this prediction is called ‘Bank Fraud’ which was found on Kaggle. The user that published the CSV on Kaggle was called ‘Orangel Mendez’ and the author was Sagar Maru. The dataset was last updated one year ago. The dataset contains 20,000 instances of 20 columns describing financial transactions such as the gender and age of consumer, the transaction date, time, amount, and type of account used to make the transaction. Each instance is a transaction. All the transaction locations are from states in India, and all currency is in Indian rupees. All transactions are dated in 2025.

Data Description: 
To identify the response variable, we looked for the column that indicates the specific event the model is designed to predict. In this context, that column is called ‘Is_Fraud’. This is a binary categorical variable where a value of 1 denotes a transaction that has been flagged as fraudulent activity, and a value of 0 represents a legitimate, non-fraudulent transaction. Because the primary objective of our project is to classify transactions into these two distinct categories, ‘isFraud’ serves as the label that guides the machine learning process during the training phase.

Final Report: Read results of the entire group. 

RF Documentation and Plot Results: See findings of my RF model. 

Raw data can be downloaded from Kaggle: 
    
    https://www.kaggle.com/datasets/orangelmendez/bank-fraud

Steps to run the R scripts: 

    Download the Bank Fraud Data CSV from the zip file. 
    Download the R scripts. There is a device focused model and general model created in both R scripts.
        ‘bank_fraud_rf_clean_script_v5.R’ is the midterm model.
        ‘bank_fraud_rf_clean_script_final.R’ is the final model that was expanded upon from the midterm. 
    Open Rstudio and restart the R session. 
    Set the working directory to the same file location as the CSV and R files.
    Load libraries: library(caret), library(randomForest), library(pROC). 
    Run the remaining script.
