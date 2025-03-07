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
    
    return dict(zip(model_names, predictions))

def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "green"
    elif aqi_value <= 100:
        return "Moderate", "orange"
    else:
        return "Bad", "red"

def get_aqi_suggestions(aqi_value, input_data):
    suggestions = []
    
    if aqi_value <= 50:
        suggestions.append("Air quality is good. Continue maintaining good practices.")
    elif aqi_value <= 100:
        suggestions.append("Air quality is moderate. Some improvements recommended:")
    else:
        suggestions.append("Air quality is poor. Urgent improvements needed:")
    
    # Add specific suggestions based on pollutant levels
    pm25, pm10, no, no2, co, so2, o3 = input_data[0]
    
    if pm25 > 35:
        suggestions.append("- Reduce PM2.5 levels by minimizing indoor combustion sources")
    if pm10 > 75:
        suggestions.append("- Lower PM10 by reducing dust through regular cleaning and air filtering")
    if no > 25 or no2 > 50:
        suggestions.append("- Decrease nitrogen oxide emissions by limiting vehicle use and improving ventilation")
    if co > 4:
        suggestions.append("- Reduce carbon monoxide by checking gas appliances and improving ventilation")
    if so2 > 40:
        suggestions.append("- Lower sulfur dioxide by reducing fossil fuel combustion")
    if o3 > 70:
        suggestions.append("- Decrease ozone levels by limiting outdoor activities during peak hours")
    
    if len(suggestions) == 1:
        suggestions.append("- All pollutant levels are within acceptable ranges")
        
    return suggestions

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
            
            # Calculate average AQI
            avg_aqi = sum(predictions.values()) / len(predictions)
            
            # Get AQI category
            category, color = get_aqi_category(avg_aqi)
            
            # Display predictions
            st.subheader('Predictions:')
            for model, pred in predictions.items():
                st.write(f"{model}: {pred:.2f}")
            
            # Display average AQI with color
            st.markdown(f"**Average AQI: <span style='color:{color}'>{avg_aqi:.2f} ({category})</span>**", unsafe_allow_html=True)
            
            # Create a bar chart of predictions
            st.bar_chart(predictions)
            
            # Display AQI improvement suggestions
            st.subheader('Suggestions to Improve Air Quality:')
            suggestions = get_aqi_suggestions(avg_aqi, input_data)
            for suggestion in suggestions:
                st.write(suggestion)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()