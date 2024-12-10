from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Predefined API key (you can change this to any value)
API_KEY = "secret-api-key"

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# API Key Validation
def get_api_key(api_key: str = Query(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Preprocessing function
def preprocess_input(input_df):
    try:
        # Ensure input has the same columns as the training data
        columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
        
        input_df = input_df[columns]
        
        # Scale the data using StandardScaler
        scaler = StandardScaler()
        input_df_scaled = scaler.fit_transform(input_df)
        
        return input_df_scaled
    except Exception as e:
        return f"Error in preprocessing input: {str(e)}"

# Prediction function
def predict(input_data):
    try:
        # Ensure input_data is 2D (reshape if it's a single row)
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make prediction using the trained model
        prediction = model.predict(input_data)
        
        # Map the prediction output (if needed, for binary classification)
        if prediction == 1:
            return "Survived"
        else:
            return "Not Survived"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Input data model
class Passenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_female: int
    Sex_male: int
    Embarked_C: int
    Embarked_Q: int
    Embarked_S: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic Survival Prediction API"}

# Prediction endpoint with API key validation
@app.post("/predict")
def predict_survival(passenger: Passenger, api_key: str = Query(...)):
    # Validate API key
    get_api_key(api_key)
    
    # Prepare input data for prediction
    input_data = pd.DataFrame([{
        'Pclass': passenger.Pclass,
        'Age': passenger.Age,
        'SibSp': passenger.SibSp,
        'Parch': passenger.Parch,
        'Fare': passenger.Fare,
        'Sex_female': passenger.Sex_female,
        'Sex_male': passenger.Sex_male,
        'Embarked_C': passenger.Embarked_C,
        'Embarked_Q': passenger.Embarked_Q,
        'Embarked_S': passenger.Embarked_S
    }])
    
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Make the prediction
    prediction_result = predict(processed_input)
    
    return {"prediction": prediction_result}
