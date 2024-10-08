#Write a python program to implement linearregression predicting house prices based on the total rooms in a dataset ofreal estate buildings.
# Ex: - Californiahousing.csv. Determine mean squarederror for the model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Preprocess the dataset
# Select relevant features (total rooms and house prices)
data = data[data['total_rooms'] < 6000]  # Optional: filter for outliers if necessary
X = data[['total_rooms']]
y = data['median_house_value']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Optionally, you can visualize the results
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, y_pred, color='red', label='Predicted Prices')
plt.title('House Prices vs Total Rooms')
plt.xlabel('Total Rooms')
plt.ylabel('Median House Value')
plt.legend()
plt.show()
