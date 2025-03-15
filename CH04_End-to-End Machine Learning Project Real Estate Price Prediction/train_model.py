import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import pickle

# Load data
housing = pd.read_csv("housing.csv")

# Fill missing values
housing["total_bedrooms"].fillna(housing["total_bedrooms"].median(), inplace=True)

# Encode categorical feature
label_encoder = LabelEncoder()
housing['ocean_proximity'] = label_encoder.fit_transform(housing['ocean_proximity'])

# Create new features
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# Select numerical features
numerical_features = ['longitude', 'latitude', 'housing_median_age',
                      'total_rooms', 'total_bedrooms', 'population',
                      'households', 'median_income', 'rooms_per_household',
                      'bedrooms_per_room', 'population_per_household']

# Apply StandardScaler
scaler = StandardScaler()
housing[numerical_features] = scaler.fit_transform(housing[numerical_features])

# Define X and y
X = housing.drop(columns=["median_house_value"])
y = housing["median_house_value"]

# Train the model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X, y)

# Save model, scaler, and encoder
with open("california_housing_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("✅ Model, Scaler, and Encoder saved successfully!")
