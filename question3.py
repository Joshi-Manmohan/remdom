#Write a python program to classify theemails from a given dataset as a spam or not spam. (Logistic regression)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Generate Sample Data
data = {
    'email': [
        "Congratulations! You've won a lottery. Click here to claim your prize.",
        "Important update regarding your bank account.",
        "Get paid for taking surveys online!",
        "Your subscription has been renewed successfully.",
        "Free gift cards available! Click now.",
        "Dear user, your account needs verification.",
        "Limited time offer! Buy one get one free.",
        "Meeting scheduled for tomorrow at 10 AM.",
        "You have received a payment of $100.",
        "Reminder: Your appointment is tomorrow."
    ],
    'label': [
        1,  # Spam
        0,  # Not Spam
        1,  # Spam
        0,  # Not Spam
        1,  # Spam
        0,  # Not Spam
        1,  # Spam
        0,  # Not Spam
        0,  # Not Spam
        0   # Not Spam
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess the Data
X = df['email']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 3: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Example of predicting new emails
new_emails = [
    "Claim your free prize now!",
    "Let's schedule a meeting next week.",
    "Your account has been compromised."
]

# Transform and predict
new_emails_tfidf = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_tfidf)

for email, prediction in zip(new_emails, predictions):
    label = 'Spam' if prediction == 1 else 'Not Spam'
    print(f"Email: '{email}' - Prediction: {label}")
