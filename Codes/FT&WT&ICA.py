import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

channel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
window_size = 256 # Averaging band power of 2 sec
step_size = 16 # Each 0.125 sec update once
subjectList = ['02']

def compute_ica(data, n_components=40):
    ica = FastICA(n_components=n_components)
    return ica.fit_transform(data.T).T

def ICA_Processing(sub, channel, window_size, step_size, n_components=40):
    results = []
    for i in range(1):  # Assuming there are 40 trials
        with open('./DataFiles/s' + sub + '.dat', 'rb') as file:
            subject = pickle.load(file, encoding='latin1')
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = 0
            while start + window_size < data.shape[1]:
                ica_data = compute_ica(data[channel][:window_size], n_components)
                results.append((sub, i+1, ica_data, labels))
                start = start + step_size
    return results


# Collect ICA results
ica_results = []
for subject in subjectList:
    ica_results.extend(ICA_Processing(subject, channel, window_size, step_size, n_components=40))

# Plot ICA Results for the first 5 components only
num_components_to_plot = 5
for result in ica_results:
    sub, trial, ica_data, labels = result
    for i in range(min(num_components_to_plot, ica_data.shape[0])):
        plt.figure(figsize=(8, 4))
        plt.plot(ica_data[i])
        plt.title(f'ICA Result (Subject: {sub}, Trial: {trial}, Component: {i+1})')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./ICA/ICA_{sub}_trial_{trial}_component_{i+1}.png')
        plt.close()



