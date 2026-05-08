import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("house_prices.csv")

X = data[
    ["area_sqft", "bedrooms", "bathrooms", "floors", "garage_spots", "age_years"]
]

y = data["price_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

pickle.dump(model, open("house_price_model.pkl", "wb"))