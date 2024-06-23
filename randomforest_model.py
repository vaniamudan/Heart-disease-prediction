import pandas as pd
import joblib
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score,recall_score,confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import json
hd_data=pd.read_csv('Heart_Disease_Prediction.csv')
print(hd_data.info())
print(hd_data.describe())
x=hd_data.drop(["Heart Disease"],axis=1)
y=hd_data['Heart Disease']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2)
dt_rf_RS=RandomForestClassifier(criterion='entropy',max_depth=7,min_samples_leaf=5,min_samples_split=16,random_state=43)
dt_rf_RS.fit(x_train,y_train)
y_pred=dt_rf_RS.predict(x_train)
y_pred=dt_rf_RS.predict_proba(x_train)
y_pred=pd.DataFrame(y_pred,columns=['prediction_0','prediction_1'])
y_pred
print('\n')
print(pd.DataFrame(np.array([1.000000,-0.094401,0.096920,0.273053,0.220056,0.123458,0.128171,-0.402215,0.098297,0.194234,0.159774,0.356081,0.106100])).T)
print(dt_rf_RS.predict(pd.DataFrame(np.array([1.000000,-0.094401,0.096920,0.273053,0.220056,0.123458,0.128171,-0.402215,0.098297,0.194234,0.159774,0.356081,0.106100])).T))
#filename='randomforest_model.pkl'
pickle.dump(dt_rf_RS,open('randomforest_model.pkl','wb'))