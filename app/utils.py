import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(input_data):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Specify column order as used during model training
        columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
                   'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
        
        # Ensure correct column order
        input_df = input_df[columns]
        
        # Standardize input
        scaler = StandardScaler()
        input_df_scaled = scaler.fit_transform(input_df)
        
        return input_df_scaled
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")

def predict(input_data, model):
    try:
        # Ensure input is a 2D array
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return human-readable output
        return "Survived" if prediction[0] == 1 else "Not Survived"
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")
