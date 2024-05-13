import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle

# Load the EEG data for one subject and one trial
subject_data_file = './DataFiles/s02.dat'
with open(subject_data_file, 'rb') as f:
    subject_data = pickle.load(f, encoding='latin1')
    eeg_data = subject_data['data'][0]  # Extract EEG data for the first trial

# Transpose EEG data if needed
# eeg_data = eeg_data.T

num_filters = 4  # Adjust as needed

# Define the number of samples for each class
num_samples_class1 = 20  # Number of samples for class 1
num_samples_class2 = 20  # Number of samples for class 2

# Define the labels for the two classes
class_labels = np.concatenate((np.ones(num_samples_class1), -np.ones(num_samples_class2)))

# Extract the EEG data for each class
class1_indices = class_labels == 1
class2_indices = class_labels == -1

eeg_data_class1 = eeg_data[class1_indices]
eeg_data_class2 = eeg_data[class2_indices]

# Calculate the covariance matrices for each class
covariance_class1 = np.mean([np.cov(trial.T) for trial in eeg_data_class1], axis=0)
covariance_class2 = np.mean([np.cov(trial.T) for trial in eeg_data_class2], axis=0)

# Calculate the spatial filters (CSP patterns)
W, _ = np.linalg.eig(np.dot(np.linalg.inv(covariance_class1 + covariance_class2), covariance_class1))

# Select the top num_filters CSP patterns
csp_patterns = np.concatenate((W[:, :num_filters], W[:, -num_filters:]), axis=1)

# Project the EEG data onto the CSP patterns
eeg_data_csp_class1 = np.array([np.dot(csp_patterns.T, trial.T) for trial in eeg_data_class1])
eeg_data_csp_class2 = np.array([np.dot(csp_patterns.T, trial.T) for trial in eeg_data_class2])

# Plot the CSP patterns
num_channels = eeg_data.shape[1]
plt.figure(figsize=(8, 6))
for i in range(num_filters):
    plt.subplot(num_filters, 1, i+1)
    plt.plot(csp_patterns[:, i])
    plt.xlabel('Channel')
    plt.ylabel(f'CSP {i+1}')
    plt.title(f'CSP Pattern {i+1}')
plt.tight_layout()
plt.show()
