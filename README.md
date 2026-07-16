# ELLEVO-PATHWAYS-ML-INTERNSHIP-SCORE-PREDICTION-SYSTEM-(REGRESSION PROJECT)




Section A: Project Overview

This is a Machine Learning Project focused on how exam scores can be predicted by paying attention to academic factors,personal factors,family factors and socio-economic factor amongst others.Steps like Data Cleaning,Summary Stattistics,Feature Selection, encoidng and model building are used for the succesful creation of this core prediction system





Section B: Dataset Description

The dataset StudentPerformanceFactors.csv contains student records with the following features

The dataset contains 6,607 student records with 20 features spanning academic performance, family background, and personal characteristics.
:

ACADEMIC FACTORS

Hours_Studied: Number of hours spent studying per week

Attendance: Attendance percentage

Previous_Scores: Previous exam scores

Tutoring_Sessions: Number of tutoring sessions attended

Exam_Score: Target variable (exam score out of 100)

PERSONAL FACTORS

Sleep_Hours: Average hours of sleep per night

Physical_Activity: Hours of physical activity per week

Motivation_Level: Self-reported motivation (Low/Medium/High)

Gender: Male/Female

FAMILY & SOCIO-ECONOMIC FACTORS

Parental_Involvement: Level of parental involvement (Low/Medium/High)

Parental_Education_Level: Highest education level of parents

Family_Income: Family income level (Low/Medium/High)

Access_to_Resources: Access to educational resources (Low/Medium/High)

Internet_Access: Internet access at home (Yes/No)

SCHOOL & ENVIRONMENTAL FACTORS

Teacher_Quality: Perceived teacher quality (Low/Medium/High)

School_Type: Public or Private

Peer_Influence: Peer influence (Positive/Neutral/Negative)

Learning_Disabilities: Presence of learning disabilities (Yes/No)

Distance_from_Home: Distance from home to school (Near/Moderate/Far)

Extracurricular_Activities: Participation in extracurriculars (Yes/No)


Section C: Methodology

DATA CLEANING & PREPROCESSING


1. Missing Values Treatment
   
Initial Missing Value Count:

Teacher_Quality: 78 missing values

Parental_Education_Level: 90 missing values

Distance_from_Home: 67 missing values

All other columns: 0 missing values

Data Imputation Strategy: Rows with missing values were dropped to maintain data integrity, resulting in 6,378 clean records.


2. Duplicate Removal
   
Identified and removed duplicate rows based on numerical columns

Retained 6,378 unique records for analysis


3.Outlier Detection & Removal
Used Interquartile Range (IQR) method to identify outliers:

Hours_Studied: 40 outliers removed

Tutoring_Sessions: 423 outliers removed

Exam_Score: 103 outliers removed

Final dataset after outlier removal: 6,275 records


4. Data Split
   
Training Set: 70% of data

Validation Set: 15% of data

Test Set: 15% of data

Random state set to 42 for reproducibility


Section D: Summary Statistics

Key Metrics After Cleaning
Metric	Value
Total Records	6,275
Average Exam Score	~67.5
Average Hours Studied	~22 hours/week
Average Attendance	~80%
Average Sleep Hours	~7 hours/night


Section E: Model Performance

Model Evaluation on Validation Set

Model	MSE	R² Score

Linear Regression	2.13	0.796

Decision Tree	4.79	0.542

Random Forest (n=100)	2.48	0.762

XGBoost	2.69	0.743

CatBoost	2.36	0.774

Random Forest (n=300)	2.47	0.76

Random Forest (n=300, depth=100)	2.47	0.76

Best Performing Model: Linear Regression

MSE: 2.13

R² Score: 0.796

Interpretation: The model explains ~80% of the variance in exam scores, indicating strong predictive capability.
Average Previous Score	~75


Section F : Model
                                                         
├── catboost_model.pkl         # CatBoost regressor

├── random_forest_model.pkl    # Random Forest (n=100)

├── xgboost_model.pkl          # XGBoost regressor

├── linear_model.pkl           # Linear Regression

├── decision_tree_model.pkl    # Decision Tree regressor

├── rf_estimators_model.pkl    # Random Forest (n=300)

├── rf_depth_model.pkl         # Random Forest (n=300, depth=100)

Section G: Conclusion

The Student Score Prediction System successfully demonstrates the application of regression models to predict academic performance based on various student, family, and environmental factors. Key findings include:

Previous Performance is the strongestr predictor of future exam scores 

Study hours and attendance significantly have an impact on exam scores

Motivation score from resylts have a signifcant positive correlation with exam scores 

Linear Regression had the best performance among the other models which are likely complex indicating a largely linear relationship between the features and the model .

The model's R² score of 0.96 indicates high probability for succesfully predicting ecam scores.

