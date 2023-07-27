import pandas as pd
import joblib

# Step 1: Load the trained model
model = joblib.load('crop_prediction_model.joblib')

# Step 2: Prepare new input data for prediction
new_weather = 28  # Example value, replace it with the actual value
new_humidity = 60  # Example value, replace it with the actual value
new_soil_nutrients = 5.2  # Example value, replace it with the actual value

new_input = [[new_weather, new_humidity, new_soil_nutrients]]

# Step 3: Use the trained model for prediction
predicted_crop = model.predict(new_input)[0]
print(f"Predicted Crop: {predicted_crop}")
