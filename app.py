from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
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
from pydantic import BaseModel

app=FastAPI()
data=pd.read_csv('Heart_Disease_Prediction.csv')
model=joblib.load('model.joblib')


   
@app.get("/")
def read_root():
    return{"message":"welcome to the ML Model API"}

@app.post("/predict/")
def get_prediction(body: Input):
    features = pd.Series([getattr(body, f"feature{i}") for i in range(1, 14)])
    #probability = round(model.predict_proba(features)[0][1], 2)
    features=np.array(data['features']).reshape(1,-1)
    prediction=model.predict(features)
    return{"class":prediction}
    