from pydantic import BaseModel, Field, computed_field, model_validator, field_validator
from typing import Literal, Annotated, Optional
from config.city_tier import tier_1_cities,tier_2_cities

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
        
    @field_validator('city')
    @classmethod
    def normalise_city(cls, v:str) -> str:
        v = v.strip().title()
        return v
        