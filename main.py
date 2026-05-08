import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;
from sklearn.preprocessing import LabelEncoder;
import pickle;


data = pd.read_csv("house_prices.csv");
print(data.isnull().sum());

encoder = LabelEncoder();
data["locality"] = encoder.fit_transform(data["locality"]);
data["condition"] = encoder.fit_transform(data["condition"]);
data["furnished"] = encoder.fit_transform(data["furnished"]);

plt.scatter(data["area_sqft"], data["price_inr"])

plt.xlabel("Area (sq ft)")
plt.ylabel("Price (INR)")
plt.title("Area vs House Price")

plt.show()

X = data[[
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "floors",
    "garage_spots",
    "age_years"
]];
y = data["price_inr"];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);  
model = LinearRegression();
model.fit(X_train, y_train);
print("=====MODEL ACCURACY=====");  
print(model.score(X_test, y_test));

print("\n=====ENTER HOUSE DETAILS FOR PRICE PREDICTION=====");
print("Enter the following details about the house:");
area_sqft = int(input("Enter area in square feet: "));
bedrooms = int(input("Enter number of bedrooms: "));
bathrooms = int(input("Enter number of bathrooms: "));
floors = int(input("Enter number of floors: "));
garage_spots = int(input("Enter number of garage spots: "));
age_years = int(input("Enter age of the house in years: "));

sample_house = [[area_sqft, bedrooms, bathrooms, floors, garage_spots, age_years]];
prediction = model.predict(sample_house)
 
print("\n===== PREDICTED PRICE =====")

print(f"Estimated Price: ₹ {prediction[0]:,.2f}")

plt.scatter(y_test, model.predict(X_test))

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.show()

importance = model.coef_ #tells us the weight of each feature in the linear regression model
feature_names = X.columns

plt.bar(feature_names, importance)
plt.xlabel("Features")
plt.ylabel("Importance (Coefficient Value)")  
plt.title("Feature Importance")

plt.show()

print("\n===== FEATURE IMPORTANCE =====")
for feature, coef in zip(feature_names, importance):
    print(f"{feature}: {coef:.2f}")
    
pickle.dump(model, open("house_price_model.pkl", "wb"))
