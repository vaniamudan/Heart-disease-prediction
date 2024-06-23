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
data=pd.read_csv('Heart_Disease_Prediction.csv')
print(data.info())
print(data.describe())
x=data.drop(["Heart Disease"],axis=1)
y=data['Heart Disease']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2)
model=RandomForestClassifier(criterion='entropy',max_depth=7,min_samples_leaf=5,min_samples_split=16,random_state=43)
model.fit(x_train,y_train)
y_pred=model.predict(x_train)
y_pred=model.predict_proba(x_train)
y_pred=pd.DataFrame(y_pred,columns=['prediction_0','prediction_1'])
y_pred
#print('\n')
#print(pd.DataFrame(np.array([70,1,4,130,322,0,2,109,0,2.4,2,3,3])).T)
#print(model.predict(pd.DataFrame(np.array([70,1,4,130,322,0,2,109,0,2.4,2,3,3])).T))
#joblib.dump(model,'model.joblib')
pickle.dump(model,open('heartdisease_randomforest.pkl','wb'))
