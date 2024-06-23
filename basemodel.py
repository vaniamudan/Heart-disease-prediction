from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
class Input(BaseModel):
    Age: float
    Sex: float
    Chest_pain_type: float
    BP: float
    Cholesterol: float
    FBS_over_120: float
    EKG_results: float
    Max_HR: int = float
    Exercise_angina: float
    ST_depression: float
    Slope_of_ST: float
    Number_of_vessels_fluro: float
    Thallium: float