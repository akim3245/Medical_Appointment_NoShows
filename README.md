# Predicting Medical Appointment No Shows

#### -- Project Status: [Active]

## Project Intro
The purpose of this project is to build a machine learning model that can predict whether patients will miss their medical appointments. 

### Partner
* [Name of Partner organization/Government department etc..]
* Website for partner
* Partner contact: [Name of Contact], [slack handle of contact if any]
* If you do not have a partner leave this section out

### Technologies
* Python
* Pandas, Numpy, Seaborn, Matplotlib, Sk-learn
* Jupyter Notebook
* Flask
* HTML
* CSS
* Bootstrap
* etc. 

## Project Description
Data Source: https://www.kaggle.com/joniarroba/noshowappointments

Dataset contains the following features:
* Gender            
* ScheduledDay      
* AppointmentDay    
* Age               
* Neighbourhood     
* Scholarship       
* Hypertension      
* Diabetes          
* Alcoholism        
* Handicap           
* SMS_received      
* No-show      

### Feature Engineering
ScheduledDay and AppointmentDay was split into separate columns as Month, Day, and Day of the Week. 

A new column, 'WaitingDays' was created to represent the number of total days until the appointment date from the initially scheduled date. 

### Model Building
K-folds cross validation was performed on Random Forest, Logistic Regression, Decision Tree and XGBoost for model selection and XGBoost which produced the highest cross val roc_auc score of 0.73 was ultimately selected.

Training and predicting the model using XGBoost with its default parameters produced a low precision and recall score. Looking at the confusion matrix, it appears that the dataset is quite imbalanced. 

To handle the imbalanced dataset, I upsampled the minority class to match the same value count as the majority class and then trained the model on the upsampled data. The recall score on upsampled model has greatly improved but the precision score seems to have dipped lower. 

## Optimizing Hyperparameters
I am currently using grid search to find the best parameters for the model.

Next, I plan to use feature selection to remove unnecessary columns in hopes of improving the model. I will create a new dataframe consisting of only the top selected features, upsample the minority class, train on the upsampled data, and then evaluate.
