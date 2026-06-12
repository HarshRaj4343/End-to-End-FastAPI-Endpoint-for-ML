import pickle
import pandas as pd

with open('Model/model.pkl', 'rb') as f:
    model = pickle.load(f)

MODEL_VERSION = '1.0.0'
# DONE MANUALLY HERE, BUT USUALLY, IT IS DONE USING MODEL REGISTRY FEATURE OF MLFLOW

def predict_output(user_input: dict):
    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    return prediction