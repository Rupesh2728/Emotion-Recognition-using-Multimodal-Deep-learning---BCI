import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Define your data
data = np.array([
    [0.00013544178693595198, 0.48287280363392987, 0.19558625753542747, 1.0, 1.0],
    [0.00026132686241901835, 0.43688855465560267, 0.23616983217889986, 1.0, 1.0],
    [0.0003289480215700074, 0.4734329367038805, 0.2143252126640336, 1.0, 1.0],
    [0.000218585502325638, 0.43774483382434476, 0.2494669997774348, 1.0, 1.0],
    [0.0, 0.5238244557615139, 0.14749523611246929, 1.0, 0.0]
])

# Split data into features and target
X = data[:, :-2]  # Features: Activity, Mobility, Complexity
y = data[:, -2:]  # Targets: Valence, Arousal

# Reshape data for CNN input (assuming 1D convolution)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2))  # Output layer with 2 units for Valence and Arousal

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Mean Squared Error (MSE) loss and Mean Absolute Error (MAE) metric

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("Testing MSE:", loss)
print("Testing MAE:", mae)
