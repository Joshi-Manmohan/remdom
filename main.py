#Write a python program to implementclassification model on Iris data set into three species. Determine all thepossible performance metrices such as accuracy score. (Logistic regression)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Example of predicting new samples
new_samples = np.array([[5.0, 3.5, 1.6, 0.2],  # Should be Setosa
                        [6.5, 3.0, 5.2, 2.0],  # Should be Virginica
                        [5.7, 2.8, 4.1, 1.3]])  # Should be Versicolor

# Predicting new samples
new_predictions = model.predict(new_samples)

# Mapping numeric predictions to species names
species_names = iris.target_names
predicted_species = [species_names[pred] for pred in new_predictions]

for sample, species in zip(new_samples, predicted_species):
    print(f"Sample: {sample} - Predicted Species: {species}")
