import streamlit as st
import pickle
import matplotlib.pyplot as plt

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
   plt.style.use("dark_background")
   
   st.subheader(f" Estimated Price: ₹ {prediction[0]:,.2f}")
   feature_names = ["Area (sq ft)", "Bedrooms", "Bathrooms", "Floors", "Garage Spots", "Age (years)"]
   importance = model.coef_ 

   plt.bar(feature_names, importance , color="#00C853")
   plt.xlabel("Features")
   plt.ylabel("Importance (Coefficient Value)")  
   plt.title("Feature Importance")

   st.pyplot(plt)
   st.write("The bar chart above shows the importance of each feature in determining the house price. Features with higher coefficients have a greater impact on the predicted price.") 