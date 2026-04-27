import streamlit as st
import joblib 
import pandas as pd
# Title Show on Website 
st.title("🏠 House Price Prediction App")

# Saved data load 
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# sidebar input section 
st.sidebar.header("Enter House Details ")

# Area Input Box 
area = st.sidebar.number_input(
    'Area Square Feet',
    min_value=500,
    max_value=100000,
    value=1500
)

# bedrooms Input 
bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

# Bathrooms input
bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# Floors input
floors = st.sidebar.number_input(
    "Floors",
    min_value=1,
    max_value=10,
    value=3
)

# Age input
age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=50,
    value=5
)

# location_score input
location_score = st.sidebar.number_input(
    "Location_score",
    min_value=1,
    max_value=10,
    value=7
)

# # predict button 
# if st.button("Predict Price"):
     
#         # User input ko dataframe me convert
#      sample = pd.DataFrame([{
#         'Area_sqr': area,
#         'Bedrooms': bedrooms,
#         'Bathrooms': bathrooms,
#         'Floors': floors,
#         'Age': age,
#         'Location_score': location_score
#     }])

# # scaling 
# sample_sc = scaler.transform(sample)

# # preiction 
# pred = model.predict(sample_sc)

# # show Result 
# st.success(f"\n Predicted House Price = {round(pred[0] ,2)}")

if st.button("Predict Price"):
    
    # DataFrame
    sample = pd.DataFrame([{
        'Area_sqr': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'Age': age,
        'Location_score': location_score
    }])

    # Scaling
    sample_sc = scaler.transform(sample)

    # Prediction
    pred = model.predict(sample_sc)

    # Output
    st.success(f"Predicted House Price = {round(pred[0], 2)}")
