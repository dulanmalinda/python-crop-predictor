import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Step 1: Load the dataset
# Assuming you have a CSV file with columns: weather, humidity, soil_nutrients, crop
data = pd.read_csv('crop_dataset.csv')

# Step 2: Data Preprocessing
# Handle any missing values or outliers in the dataset
# For this example, let's assume the data is already clean

# Step 3: Feature Engineering
X = data[['weather', 'humidity', 'soil_nutrients']]
y = data['crop']

# Step 4: Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection and Training
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Save the trained model to a file
joblib.dump(model, 'crop_prediction_model.joblib')
