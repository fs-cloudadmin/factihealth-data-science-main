# Hospital Readmission Prediction

## Introduction
Hospital readmission prediction is a crucial task in healthcare management aimed at identifying patients at risk of being readmitted to the hospital within a certain period after their initial discharge. This project utilizes machine learning techniques to develop predictive models that can assist healthcare providers in early identification and intervention, ultimately improving patient outcomes and reducing healthcare costs.

## Data Source
The dataset used for this project is sourced from the MIMIC-IV database version 2.2, available on PhysioNet. The dataset contains de-identified electronic health records from patients admitted to the Beth Israel Deaconess Medical Center from 2011 to 2019. The dataset is loaded into a Redshift database for further analysis and processing.

## Preprocessing Steps
1. Merge Admission Data with Diagnosis Data
2. Merge Admission Data with Chart Events
3. Group Patient Label Data with Hospital Admissions
4. Recategorize Admission Type and Race
5. Mapping of ICD Codes to Categories

## Rules Applied
- Recategorize Admission Type
- Recategorize Race

## Exploratory Data Analysis (EDA)
Exploratory data analysis is conducted to understand the distribution of diagnoses across different categories, derived based on the mapping of ICD codes.

## Model Building
Random Forest Classifier is used for building predictive models to predict readmissions at different time intervals (30 days, 60 days, and 365 days).

## Model Evaluation
The performance of the models is evaluated using standard classification metrics such as precision, recall, F1-score, and accuracy.

| Time Interval | Precision (0) | Recall (0) | F1-score (0) | Precision (1) | Recall (1) | F1-score (1) | Accuracy |
|---------------|---------------|------------|--------------|---------------|------------|--------------|----------|
| 30 days       | 0.90          | 0.99       | 0.94         | 0.97          | 0.74       | 0.84         | 0.92     |
| 60 days       | 0.91          | 0.97       | 0.94         | 0.95          | 0.87       | 0.91         | 0.92     |
| 365 days      | 0.91          | 0.99       | 0.95         | 0.98          | 0.76       | 0.85         | 0.93     |

## Deployment
The trained models are deployed using pickle files, facilitating easy integration into production systems for real-time prediction of readmission risks.

## Files Used
1. Readmission_Data_Exploration_Retrain
2. EDA_Readmissions.ipynb
3. ModelBuilding.ipynb
4. Readmission_Model_Random_100.ipynb

## Conclusion
This project demonstrates the application of machine learning techniques for hospital readmission prediction, providing valuable insights into patient risk factors and enabling proactive care management.
