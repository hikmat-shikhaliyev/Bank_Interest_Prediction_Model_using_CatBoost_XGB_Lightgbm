# Interest Prediction Model
This repository contains Python code for building, optimizing, and deploying a machine learning model to predict customer interest in a banking product based on historical data. The model uses various features such as age, vintage, gender, occupation, channel code, credit product, and activity status to make predictions. Different machine learning algorithms, including CatBoost, XGBoost, and LightGBM, are explored and optimized to achieve the best performance. The repository also includes code for model evaluation, feature analysis, and visualization of results.

# Prerequisites
Before running the code, ensure you have the following libraries installed:
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Statsmodels
CatBoost
XGBoost
LightGBM

# Data Preprocessing
Reads the input data from interest_prediction_bank.csv.
Removes unnecessary columns (ID, Region_Code) and handles missing values.
Encodes categorical variables (Gender, Occupation, Channel_Code, Credit_Product, Is_Active) using one-hot encoding.
Splits the data into features (X) and target (y) variables.
Divides the data into training and testing sets (80% training, 20% testing).

# Model Building and Evaluation
Utilizes various machine learning algorithms including CatBoost, XGBoost, and LightGBM.
Performs hyperparameter tuning using Randomized Search Cross Validation.
Evaluates model performance using ROC-AUC score and Gini coefficient.
Compares and analyzes models' performance to select the best one.

# Feature Analysis
Computes the correlation of each feature with the target variable (Is_interested).
Conducts variance inflation factor (VIF) analysis to identify multicollinearity.
Generates box plots to identify outliers in numeric features (Age and Vintage).

# Model Deployment
Uses the best-performing model for deployment (univariate_lightgbm) to make predictions on new data (interest_prediction_bank_test_set.csv).
Outputs the predicted probabilities of customer interest in the banking product.

# Results
The repository includes trained machine learning models, evaluation results, and visualizations, providing a comprehensive understanding of the predictive model's performance.
