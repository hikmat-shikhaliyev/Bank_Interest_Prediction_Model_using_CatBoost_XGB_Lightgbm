#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
sns.set()


# In[78]:


data=pd.read_csv(r'C:\Users\ASUS\Downloads\interest_prediction_bank.csv')
data


# In[79]:


data.describe(include='all')


# In[80]:


data.drop(['ID', 'Region_Code'], axis=1, inplace=True)


# In[81]:


data.isnull().sum()


# In[82]:


data['Credit_Product']=data['Credit_Product'].fillna(value=data['Credit_Product'].mode()[0])


# In[83]:


data.isnull().sum()


# In[84]:


data.corr()['Is_interested']


# In[85]:


data.drop('Avg_Account_Balance', axis=1, inplace=True)


# In[86]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=data[['Age', 'Vintage']]
vif=pd.DataFrame()
vif['VIF']=[variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features']=variables.columns
vif


# In[87]:


for i in data[['Age', 'Vintage']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[88]:


from catboost import CatBoostClassifier


# In[89]:


from xgboost import XGBClassifier


# In[90]:


from lightgbm import LGBMClassifier


# In[91]:


data.head()


# In[92]:


Input=data.drop('Is_interested', axis=1)
Output=data['Is_interested']


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(Input, Output, test_size=0.2, random_state=42)


# In[94]:


def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test= roc_score_test*2-1
    
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    

    print('Model Performance')

    print('Gini Score for Test:', gini_score_test*100)
    print('Gini Score for Train:', gini_score_train*100)


# In[95]:


catboost_base_model = CatBoostClassifier(cat_features=['Gender', 'Occupation', 'Channel_Code', 'Credit_Product', 'Is_Active'])


# In[96]:


catboost_base_model.fit(X_train, y_train)


# In[97]:


result = evaluate(catboost_base_model, X_test, y_test)


# In[98]:


new_data=data.copy()


# In[99]:


new_data


# In[100]:


new_data=pd.get_dummies(new_data, drop_first=True)


# In[101]:


new_data


# In[102]:


X=new_data.drop('Is_interested', axis=1)
y=new_data['Is_interested']


# In[103]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[104]:


xgboost_base_model=XGBClassifier()
xgboost_base_model.fit(X_train, y_train)


# In[105]:


result = evaluate(xgboost_base_model, X_test, y_test)


# In[106]:


lightgbm_base_model=LGBMClassifier()
lightgbm_base_model.fit(X_train, y_train)


# In[107]:


result = evaluate(lightgbm_base_model, X_test, y_test)


# In[108]:


catboost_with_dummies = CatBoostClassifier()
catboost_with_dummies.fit(X_train, y_train)


# In[109]:


result = evaluate(catboost_with_dummies , X_test, y_test)


# In[110]:


#Hyperparameter Tuning for lightgbm

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 10],
    'num_leaves': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]    
}

param_distributions


# In[111]:


lightgbm_randomized=RandomizedSearchCV(lightgbm_base_model, param_distributions=param_distributions, n_iter=10, cv=5, n_jobs=-1, random_state=42)
lightgbm_randomized.fit(X_train, y_train)


# In[112]:


print('Best hyperparameters for the lightgbm:', lightgbm_randomized.best_params_)


# In[113]:


optimized_lightgbm=lightgbm_randomized.best_estimator_


# In[114]:


result=evaluate(optimized_lightgbm, X_test, y_test)


# In[115]:


#Hyperparameter Tuning for xgboost
param_distributions = {
    
    'n_estimators': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 10],
    'subsample': np.linspace(0.5, 1, num=6),
    'colsample_bytree': np.linspace(0.5, 1, num=6),
    'gamma': [0,1,5,10]
    
}

param_distributions


# In[116]:


xgboost_randomized = RandomizedSearchCV(xgboost_base_model, param_distributions=param_distributions, n_iter=10, cv=5, n_jobs=-1, random_state=42)


# In[117]:


xgboost_randomized.fit(X_train, y_train)


# In[118]:


print('Best hyperparameters for XGBoost:', xgboost_randomized.best_params_)


# In[119]:


optimized_xgboost=xgboost_randomized.best_estimator_


# In[120]:


result = evaluate(optimized_xgboost, X_test, y_test)


# In[121]:


#Hyperparameter Tuning for catboost
param_distributions = {
    
    'iterations': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'depth': [3, 5, 7, 9],
    'l2_leaf_reg': np.linspace(2, 30, num=7)
    
}

param_distributions


# In[122]:


catboost_randomized=RandomizedSearchCV(catboost_with_dummies, param_distributions=param_distributions, cv=5, n_iter=10, random_state=42)


# In[123]:


catboost_randomized.fit(X_train, y_train)


# In[124]:


optimized_catboost=catboost_randomized.best_estimator_


# In[125]:


result = evaluate(optimized_catboost, X_test, y_test)


# ## Stacking Model

# In[126]:


from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier


# In[127]:


base_classifiers = [
    lightgbm_base_model,
    optimized_xgboost,
    RandomForestClassifier(random_state=42),
    catboost_with_dummies
]


# In[128]:


meta_classifier = lightgbm_base_model


# In[129]:


stacking_classifier = StackingCVClassifier(classifiers=base_classifiers,
                                           meta_classifier=meta_classifier,
                                           cv=5,
                                           use_probas=True,
                                           use_features_in_secondary=True,
                                           verbose=1,
                                           random_state=42)


# In[130]:


stacking_classifier.fit(X_train, y_train)


# In[131]:


result=evaluate(stacking_classifier, X_test, y_test)


# In[132]:


from sklearn.metrics import roc_curve
y_prob = lightgbm_base_model.predict_proba(X_test)[:, 1]

roc_prob= roc_auc_score(y_test, y_prob)
gini = 2*roc_prob-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Auc = %0.2f)' % roc_prob)
plt.plot(fpr, tpr, label='(Gini = %0.2f)' % gini)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[133]:


variables= []
train_Gini=[]
test_Gini=[]

for i in X_train.columns:
    X_train_single=X_train[[i]]
    X_test_single=X_test[[i]]
    
    lightgbm_base_model.fit(X_train_single, y_train)
    y_prob_train_single=lightgbm_base_model.predict_proba(X_train_single)[:, 1]
    
    
    roc_prob_train=roc_auc_score(y_train, y_prob_train_single)
    gini_prob_train=2*roc_prob_train-1
    
    
    lightgbm_base_model.fit(X_test_single, y_test)
    y_prob_test_single=lightgbm_base_model.predict_proba(X_test_single)[:, 1]
    
    
    roc_prob_test=roc_auc_score(y_test, y_prob_test_single)
    gini_prob_test=2*roc_prob_test-1
    
    
    variables.append(i)
    train_Gini.append(gini_prob_train)
    test_Gini.append(gini_prob_test)
    

df = pd.DataFrame({'Variable': variables, 'Train Gini': train_Gini, 'Test Gini': test_Gini})

df= df.sort_values(by='Test Gini', ascending=False)

df   


# In[134]:


new_data.columns


# In[135]:


X=new_data[['Age', 'Vintage', 'Gender_Male',
       'Occupation_Salaried', 'Occupation_Self_Employed', 'Channel_Code_X2',
       'Channel_Code_X3', 'Credit_Product_Yes',
       'Is_Active_Yes']]
y=new_data['Is_interested']


# In[136]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[137]:


univariate_lightgbm=LGBMClassifier()


# In[138]:


univariate_lightgbm.fit(X_train, y_train)


# In[139]:


result=evaluate(univariate_lightgbm, X_test, y_test)


# ## Deployment

# In[140]:


test_data=pd.read_csv(r'C:\Users\ASUS\Downloads\interest_prediction_bank_test_set.csv')
test_data


# In[141]:


X.columns


# In[142]:


test_data=test_data[['Age', 'Vintage', 'Gender', 'Occupation', 'Channel_Code','Credit_Product', 'Is_Active']]


# In[143]:


test_data=pd.get_dummies(test_data, drop_first=True)


# In[144]:


test_data


# In[145]:


test_data=test_data[['Age', 'Vintage', 'Gender_Male', 'Occupation_Salaried',
       'Occupation_Self_Employed', 'Channel_Code_X2', 'Channel_Code_X3',
       'Credit_Product_Yes', 'Is_Active_Yes']]


# In[146]:


test_data


# In[147]:


X


# In[148]:


test_data['Prediction']=univariate_lightgbm.predict_proba(test_data)[:,1]


# In[149]:


test_data

