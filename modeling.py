# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:26:32 2022

@author: biyin
"""
from sklearn.metrics import fbeta_score, make_scorer
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from math import radians, sin, cos, sqrt, asin

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import XGBRFClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, plot_confusion_matrix, f1_score, roc_auc_score, plot_roc_curve, roc_curve, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

from IPython.display import Image
from IPython.core.display import HTML 

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


#Function to get results
def analyse_model_performance(model, X_train_values, y_train_values, X_test_values, y_test_values):
    
    
    # Get predictions
    preds = model.predict(X_test_values)

    # Save confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_test_values, preds).ravel()

    dict_scores = {
                   'cross_val_score': model.best_score_, 
                   'train_auc_score': model.score(X_train_values, y_train_values),
                   'test_auc_score': model.score(X_test_values, y_test_values),
                   'train_accuracy_score': accuracy_score(y_train_values, model.predict(X_train_values)),
                   'test_accuracy_score': accuracy_score(y_test_values, model.predict(X_test_values)),               
                   'sensitivity': tp / (tp + fn),
                   'specificity': tn / (tn + fp)}

    print(f"optimal Parameters: \n{model.best_params_}\n")
    print(f"Cross Validation Score: \n{dict_scores['cross_val_score']}\n")
    print(f"Train AUC Score: \n{dict_scores['train_auc_score']}\n")
    print(f"Test AUC SCore: \n{dict_scores['test_auc_score']}\n")
    print(f"Train Accuracy Score: \n{dict_scores['train_accuracy_score']}\n")
    print(f"Test Accuracy Score: \n{dict_scores['test_accuracy_score']}\n")
    print(f"Sensitivity: \n{dict_scores['sensitivity']}\n")    
    print(f"Specificity: \n{dict_scores['specificity']}\n")
    
    return dict_scores


#Modeling
df = pd.read_csv('coord_export_combined.csv')

X = df.iloc[:, 1: 133]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3 ,random_state = 42, stratify= y)

# Scale the features
ss = StandardScaler()

X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

#Random Forest Classifier
pipe = Pipeline([('rf', RandomForestClassifier())])

#Setting Pipe Parameters
pp = {
                     'rf__n_estimators': [100, 150, 200],
                     'rf__max_depth': [None,4],
                     'rf__random_state': [123]}

scoring={'AUC': 'roc_auc', 'Accuracy':make_scorer(accuracy_score)}

#gridsearchcv
gs_rf = GridSearchCV(pipe, param_grid = pp, cv=5, scoring=scoring, refit='AUC') 

#results
gs_rf.fit(X_train, y_train)
score_dict = analyse_model_performance(gs_rf, X_train, y_train, X_test, y_test)

# saving our model # model - model , filename-model_jlib
joblib.dump(gs_rf , 'RandomForest')

