from fastapi import FastAPI, HTTPException, Path, Query
import json
from pydantic import BaseModel, Field, computed_field, model_validator, field_validator
from typing import Literal, Annotated, Optional
from schema.schema import Schema
from Model.predict import predict_output,model,MODEL_VERSION
from fastapi.responses import JSONResponse
from schema.response_model import PredictionResponse

app = FastAPI() 
# In pydantic v2 , every @computed_field must have a return type.

@app.get("/")
def home():
    return {'message':'Insurance Premium Prediction API'}

# mainly for aws health checkup
@app.get("/health")
def health_check():
    return {
        'status':'OK',
        'version':MODEL_VERSION
    }

@app.post("/predict",response_model=PredictionResponse)
def predict(details: Schema):
    user_input ={
        'bmi': details.bmi,
        'age_group': details.age_group,
        'lifestyle_risk': details.lifestyle_risk,
        'city_tier': details.city_tier,
        'income_lpa': details.income_lpa,
        'occupation': details.occupation
    }
    try:
        prediction = predict_output(user_input)

        return JSONResponse(status_code=200, content={'predicted_category': prediction})
    except Exception as e:
        return JSONResponse(status_code=500,content=str(e))
