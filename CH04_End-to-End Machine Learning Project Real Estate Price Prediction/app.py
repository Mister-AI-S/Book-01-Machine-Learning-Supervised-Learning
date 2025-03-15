# 1). Import Required Libraries
import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 2). Load model, scaler, and encoder
try:
    model = pickle.load(open("california_housing_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model or scaler file missing. Train the model first!")

# 3). Function to preprocess input data
def preprocess_data(df):
    """Preprocesses user input data using saved scaler & encoder."""
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    # Encode categorical variable using trained encoder
    df["ocean_proximity"] = label_encoder.transform(df["ocean_proximity"])

    # Scale numerical features
    num_features = ["longitude", "latitude", "housing_median_age", "total_rooms",
                    "total_bedrooms", "population", "households", "median_income",
                    "rooms_per_household", "bedrooms_per_room", "population_per_household"]
    df[num_features] = scaler.transform(df[num_features])

    return df

# -------------- Streamlit UI --------------------------------

# 4). Title & Description 
st.title("🏡 California Housing Price Prediction")
st.write("Click on the map to select a location.")

# 5). MAP INPUT ---
st.header("📌 Select a Location on Map (California Only)")

# Default location (California center)
default_location = [37, -119.5]

# Create a folium map
m = folium.Map(location=default_location, zoom_start=5)

# Add a click event to get coordinates
m.add_child(folium.LatLngPopup())

# Display map in Streamlit
map_data = st_folium(m, height=400, width=700)

# Extract clicked location
if map_data and map_data.get("last_clicked"):
    latitude, longitude = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
else:
    latitude, longitude = default_location

st.write(f"**📌 Selected Location:** Longitude = `{longitude:.4f}`, Latitude = `{latitude:.4f}`")

# 6). Feature Inputs ---
housing_median_age = st.slider("⏳ Housing Median Age", 1, 50, 5)
total_rooms = st.number_input("🚪 Total Rooms", value=2000)
total_bedrooms = st.number_input("🛏️ Total Bedrooms", value=500)
population = st.number_input("👨‍👩‍👧‍👦 Population", value=1000)
households = st.number_input("🏘️ Households", value=400)
median_income = st.number_input("💰 Median Income", value=3.5)
ocean_proximity = st.selectbox("🌊  Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# Convert input into DataFrame
input_data = pd.DataFrame([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                            population, households, median_income, ocean_proximity]],
                          columns=["longitude", "latitude", "housing_median_age", "total_rooms",
                                   "total_bedrooms", "population", "households", "median_income", "ocean_proximity"])

# 7). Prediction Section ---
if st.button("Predict Price"):
    processed_input = preprocess_data(input_data)
    prediction = model.predict(processed_input)[0]
    st.success(f"🏠 Predicted Median House Value: **${prediction:,.2f}**")


# 8). UPDATE MODEL SECTION ---
st.header("🔄💰 Update Model with Real Price")
st.write("Enter the actual median house value to improve the model.")

# User enters actual price
actual_median_house_value = st.number_input("✅💰Actual Median House Value", value=150000, format="%d")

# Update button
if st.button("Update Model"):

    # Load existing dataset
    df_new = pd.read_csv("housing.csv") 

    # Add actual price to input data
    input_data["median_house_value"] = actual_median_house_value

     # Append new data
    df_new = pd.concat([df_new, input_data], ignore_index=True)
    
    # Save updated dataset
    df_new.to_csv("housing.csv", index=False)

    # Feature engineering
    df_new["rooms_per_household"] = df_new["total_rooms"] / df_new["households"]
    df_new["bedrooms_per_room"] = df_new["total_bedrooms"] / df_new["total_rooms"]
    df_new["population_per_household"] = df_new["population"] / df_new["households"]

    # Encode categorical variable using the trained encoder
    df_new["ocean_proximity"] = label_encoder.transform(df_new["ocean_proximity"])

    # Select features and target
    X_new = df_new.drop(columns=["median_house_value"])
    y_new = df_new["median_house_value"]

    # Scale numerical features using the trained scaler
    num_features = ["longitude", "latitude", "housing_median_age", "total_rooms",
                    "total_bedrooms", "population", "households", "median_income",
                    "rooms_per_household", "bedrooms_per_room", "population_per_household"]
    X_new[num_features] = scaler.transform(X_new[num_features])

    # Train model with new data
    model.fit(X_new, [y_new])  # Online learning: training on single instance

    # Save updated model
    with open("california_housing_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open("label_encoder.pkl", "wb") as encoder_file:
        pickle.dump(label_encoder, encoder_file)

    st.success("✅ Model updated successfully with new data!")