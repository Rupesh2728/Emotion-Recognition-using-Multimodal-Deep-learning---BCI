import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft

# Assuming you have loaded your dataset and extracted EEG data
# Here's a mock example with random EEG data for demonstration purposes
# Replace this with your actual data loading and preprocessing
num_subjects = 32
num_trials = 40
num_channels = 40
num_samples_per_trial = 1000

# Mock EEG data
eeg_data = np.random.rand(num_subjects, num_trials, num_channels, num_samples_per_trial)

# Choose a sample subject, trial, and channel for demonstration
subject_idx = 0
trial_idx = 0
channel_idx = 0
eeg_sample = eeg_data[subject_idx, trial_idx, channel_idx]

# Fourier Transform
fft_result = fft(eeg_sample)
freqs = np.fft.fftfreq(len(eeg_sample))

# Wavelet Transform
wavelet = 'db1'  # Using Daubechies wavelet for demonstration
coeffs = pywt.wavedec(eeg_sample, wavelet)
levels = len(coeffs)

# Plotting
plt.figure(figsize=(12, 6))

# Plot EEG Data
plt.subplot(3, 1, 1)
plt.plot(eeg_sample)
plt.title('EEG Data')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Plot Fourier Transform
plt.subplot(3, 1, 2)
plt.plot(freqs, np.abs(fft_result))
plt.title('Fourier Transform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# Plot Wavelet Transform
plt.subplot(3, 1, 3)
plt.plot(coeffs[0])  # Plot the approximation coefficients (cA)
for i in range(1, levels):
    plt.plot(coeffs[i])
plt.title('Wavelet Transform')
plt.xlabel('Scale')
plt.ylabel('Coefficient')

plt.tight_layout()
plt.show()
