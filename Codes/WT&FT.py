import numpy as np
import pickle as pickle
import pywt
import os
import time
import matplotlib.pyplot as plt

channel = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31] #14 Channels chosen to fit Emotiv Epoch+
window_size = 256 #Averaging band power of 2 sec
step_size = 16 #Each 0.125 sec update once
# subjectList = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']
subjectList = ['01','02','03','04','05','06','07','08','09','10']

def compute_fourier(data, channel, window_size):
    meta_data_fourier = []
    for j in channel:
        X = data[j][:window_size]
        Y_fourier = np.fft.fft(X)
        Y_fourier = np.abs(Y_fourier[:window_size//2])  # Taking only positive frequencies
        meta_data_fourier.append(Y_fourier)
    return np.hstack(meta_data_fourier)

def compute_wavelet(data, channel, window_size, wavelet='db4', level=4):
    meta_data_wavelet = []
    for j in channel:
        X = data[j][:window_size]
        coeffs = pywt.wavedec(X, wavelet, level=level)
        for coeff in coeffs:
            meta_data_wavelet.append(coeff)
    return np.hstack(meta_data_wavelet)


def Fourier_Processing(sub, channel, window_size, step_size):
    meta = []
    with open('./DataFiles/s' + sub + '.dat', 'rb') as file:
        subject = pickle.load(file, encoding='latin1')
        for i in range(2):  # Assuming there are 40 trials
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = 0
            while start + window_size < data.shape[1]:
                meta_data_fourier = compute_fourier(data, channel, window_size)

                meta.append((meta_data_fourier, labels))

                # Plotting the results
                plt.figure(figsize=(8, 4))
                plt.plot(meta_data_fourier)
                plt.title('Fourier Transform Results (Subject: ' + sub + ', Trial: ' + str(i + 1) + ')')
                plt.xlabel('Frequency Bin')
                plt.ylabel('Amplitude')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('./FT&WT/Fourier/Fourier_' + sub + '_trial_' + str(i + 1) + '.png')
                plt.close()

                start = start + step_size


def Wavelet_Processing(sub, channel, window_size, step_size, wavelet='db4', level=4):
    meta = []
    with open('./DataFiles/s' + sub + '.dat', 'rb') as file:
        subject = pickle.load(file, encoding='latin1')
        for i in range(1, 2):  # Assuming there are 40 trials, here considering only one trail
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = 0
            while start + window_size < data.shape[1]:
                meta_data_wavelet = compute_wavelet(data, channel, window_size, wavelet, level)

                meta.append((meta_data_wavelet, labels))

                # Plotting the results
                plt.figure(figsize=(8, 4))
                plt.plot(meta_data_wavelet)
                plt.title('Wavelet Transform Results (Subject: ' + sub + ', Trial: ' + str(i + 1) + ')')
                plt.xlabel('Wavelet Coefficient')
                plt.ylabel('Amplitude')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('./FT&WT/Wavelet/Wavelet_' + sub + '_trial_' + str(i + 1) + '.png')
                plt.close()

                start = start + step_size


# for subjects in subjectList:
#     Fourier_Processing(subjects, channel, window_size, step_size)

for subjects in subjectList:
    Wavelet_Processing(subjects, channel, window_size, step_size, wavelet='db4', level=4)

