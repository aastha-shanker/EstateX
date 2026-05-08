import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

# Load the trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

st.title("🏠 EstateX")
st.subheader("Smart House Price Prediction System") 

st.write("Welcome to EstateX! Enter the details of the house below to get an estimated price."
         "Our model is trained on a comprehensive dataset of house prices and features, ensuring accurate predictions." 
         "Whether you're a buyer or a seller, EstateX can help you navigate the complex world of real estate transactions with confidence.")

#input the values for the house features
area_sqft = st.number_input("Enter area in square feet", min_value=100, max_value=10000, step=50)
bedrooms = st.number_input("Enter number of bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Enter number of bathrooms", min_value=1, max_value=10, step=1)
floors = st.number_input("Enter number of floors", min_value=1, max_value=5, step=1)
garage_spots = st.number_input("Enter number of garage spots", min_value=0, max_value=5, step=1)
age_years = st.number_input("Enter age of the house in years", min_value=0, max_value=100, step=1)    

if st.button("Predict Price"):
   sample_house = [[
        area_sqft,
        bedrooms,
        bathrooms,
        floors,
        garage_spots,
        age_years
    ]]
   prediction = model.predict(sample_house)
   
   
   st.subheader(f" Estimated Price: ₹ {prediction[0]:,.2f}")
   feature_names = ["Area (sq ft)", "Bedrooms", "Bathrooms", "Floors", "Garage Spots", "Age (years)"]
   importance = model.coef_

   df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
   })

   df = df.sort_values("Importance")

   chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("Importance:Q"),
    y=alt.Y("Feature:N", sort="-x"),
    color=alt.condition(
        alt.datum.Importance > 0,
        alt.value("#00C853"),
        alt.value("#FF5252")
    )
   ).properties(
    title="Feature Importance"
   )

   st.altair_chart(chart, use_container_width=True)

   st.write("The bar chart shows how each feature impacts house price prediction.")
   st.write("Features with positive importance increase the predicted price, while those with negative importance decrease it.")
   st.write("Thank you for using EstateX! We hope our smart house price prediction system helps you make informed real estate decisions.")