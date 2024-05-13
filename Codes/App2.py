import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Define your data
df = pd.read_csv("./features/features_normalized_2.csv", header=None)
data = np.array(df)

# Split data into features and target
X = data[:, :-2]  # Features: Activity, Mobility, Complexity
y = data[:, -2:]  # Targets: Valence, Arousal

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# Define a threshold for classifying the predicted values
threshold = 0.5

# Predict on the testing set
y_pred = model.predict(X_test)

# Convert predicted values to binary based on the threshold
y_pred_binary = np.where(y_pred >= threshold, 1, 0)

# Convert true values to binary based on the threshold
y_test_binary = np.where(y_test >= threshold, 1, 0)

# Calculate accuracy
accuracy = np.mean(y_pred_binary == y_test_binary)
print("Testing accuracy:", accuracy)
