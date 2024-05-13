import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

# Define parameters
wavelet_type = 'morl'
num_scales = 1
min_freq = 0.5
max_freq = 30
fs = 128  # Assuming the sampling frequency of the DEAP dataset is 128 Hz

# Load DEAP data for one subject (e.g., subject 02)
subject_folder = './DataFiles/'
subject_files = os.listdir(subject_folder)
subject_data = None
for file in subject_files:
    if file.startswith('s02'):
        with open(os.path.join(subject_folder, file), 'rb') as f:
            subject_data = np.load(f, encoding='latin1', allow_pickle=True)
            break

if subject_data is None:
    print("Subject data not found.")
    exit()

# Extract data and labels
data = subject_data['data']
labels = subject_data['labels']

# Select one trial (e.g., first trial)
trial_index = 0
eeg_data = data[trial_index]

# Perform continuous wavelet transform for each channel
power_spectrum_all_channels = []
frequencies_all_channels = []
for i in range(eeg_data.shape[0]):
    # Perform continuous wavelet transform
    coefficients, frequencies = pywt.cwt(eeg_data[i, :], np.logspace(np.log10(min_freq), np.log10(max_freq), num_scales), wavelet_type, sampling_period=1/fs)

    # Compute the wavelet power spectrum
    power_spectrum = np.abs(coefficients)**2

    # Store the results for this channel
    power_spectrum_all_channels.append(power_spectrum)
    frequencies_all_channels.append(frequencies)

# Combine the results for all channels
mean_power_spectrum = np.mean(np.array(power_spectrum_all_channels), axis=0)
mean_frequencies = np.mean(np.array(frequencies_all_channels), axis=0)

# Plot the mean wavelet power spectrum
plt.figure(figsize=(10, 6))
plt.imshow(mean_power_spectrum, extent=[0, eeg_data.shape[1], mean_frequencies[-1], mean_frequencies[0]],
           aspect='auto', cmap='jet', origin='lower')
plt.colorbar(label='Power')
plt.xlabel('Time (samples)')
plt.ylabel('Frequency (Hz)')
plt.title('Mean Wavelet Power Spectrum (Subject: 02, Trial: 1)')
plt.savefig('./FT&WT/WT_s02' + '.png')
plt.show()
