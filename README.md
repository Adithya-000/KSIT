AQI Prediction App
A simple Streamlit app to predict the Air Quality Index (AQI) using pre-trained machine learning models based on pollutant levels.
What It Does
Takes input for pollutants: PM2.5, PM10, NO, NO2, CO, SO2, O3.
Predicts AQI using Linear Regression, Lasso, Ridge, and Decision Tree models.
Shows predictions as text and a bar chart.
Requirements
Python 3.7+
Libraries: streamlit, pandas, numpy, scikit-learn, joblib
Model files: linear_model.pkl, lasso_model.pkl, ridge_model.pkl, dt_model.pkl
Setup
Install Dependencies:
bash
pip install streamlit pandas numpy scikit-learn joblib
Add Model Files:
Place the .pkl files in the same directory as aqi_app.py.
(If you don’t have them, you’ll need to train and save the models separately.)
How to Run
Save the code as aqi_app.py.
Run the app:
bash
streamlit run aqi_app.py
Open http://localhost:8501 in your browser.
Usage
Enter pollutant levels in the input fields.
Click "Predict AQI" to see the results.
