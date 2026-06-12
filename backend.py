from fastapi import FastAPI, HTTPException, Path, Query
import json
from pydantic import BaseModel, Field, computed_field, model_validator
from typing import Literal, Annotated, Optional
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
# import the ml model

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
app = FastAPI() 
# In pydantic v2 , every @computed_field must have a return type.

tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

class Schema(BaseModel):

    income_lpa: Annotated[int, Field(..., description='Income in lakhs per annum')]
    smoker: Annotated[bool, Field(..., description='True if you smoke, else False')]
    occupation: Annotated[Literal['retired','freelancer','student','government_job', 'business_owner','unemployed','private_job'], Field(..., description='Occupation?')]
    city: Annotated[str, Field(..., description='City where you are living')]
    age: Annotated[int, Field(..., gt=0, lt=120, description='Age')]
    height: Annotated[float, Field(..., gt=0, description='Height in mtrs')]
    weight: Annotated[float, Field(..., gt=0, description='Weight in kgs')]

    @computed_field
    @property
    def bmi(self) -> float:
        bmi = round(self.weight/(self.height**2),2)
        return bmi
    
    @computed_field
    @property
    def age_group(self)->str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"
    
    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        return "low"
    
    
    @computed_field
    @property
    def city_tier(self)->int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
        
@app.post("/predict")
def predict(details: Schema):
    input_df = pd.DataFrame([{
        'bmi': details.bmi,
        'age_group': details.age_group,
        'lifestyle_risk': details.lifestyle_risk,
        'city_tier': details.city_tier,
        'income_lpa': details.income_lpa,
        'occupation': details.occupation
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200, content={'predicted_category': prediction})
