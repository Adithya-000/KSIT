import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
import joblib

def load_models():
    # Load the trained models
    reg1 = joblib.load('linear_model.pkl')
    reg2 = joblib.load('lasso_model.pkl')
    reg3 = joblib.load('ridge_model.pkl')
    reg4 = joblib.load('dt_model.pkl')
    return reg1, reg2, reg3, reg4

def predict_aqi(input_data, models):
    # Make predictions using all models
    predictions = []
    model_names = ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Decision Tree']
    
    for model in models:
        pred = model.predict(input_data)
        predictions.append(pred[0])
    
    return predictions

def classify_aqi(aqi):
    if aqi <= 50:
        return "Perfect"
    elif aqi <= 100:
        return "Moderate"
    else:
        return "High"

def main():
    st.title('Air Quality Index Prediction')
    st.write("""
    This application predicts the Air Quality Index (AQI) based on various pollutant levels.
    Please enter the values for different parameters below.
    """)

    # Create input fields
    pm25 = st.number_input('PM2.5 Level', min_value=0.0, max_value=500.0, value=50.0)
    pm10 = st.number_input('PM10 Level', min_value=0.0, max_value=500.0, value=100.0)
    no = st.number_input('NO Level', min_value=0.0, max_value=500.0, value=20.0)
    no2 = st.number_input('NO2 Level', min_value=0.0, max_value=500.0, value=40.0)
    co = st.number_input('CO Level', min_value=0.0, max_value=500.0, value=1.0)
    so2 = st.number_input('SO2 Level', min_value=0.0, max_value=500.0, value=30.0)
    o3 = st.number_input('O3 Level', min_value=0.0, max_value=500.0, value=45.0)

    if st.button('Predict AQI'):
        # Prepare input data
        input_data = np.array([[pm25, pm10, no, no2, co, so2, o3]])
        
        try:
            # Load models
            models = load_models()
            
            # Get predictions
            predictions = predict_aqi(input_data, models)
            
            # Display predictions
            st.subheader('Predictions:')
            for model, (pred, category) in predictions.items():
                st.write(f"{model}: {pred:.2f} ({category})")
            
            # Create a bar chart of predictions
            st.bar_chart({model: pred for model, (pred, _) in predictions.items()})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
