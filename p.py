import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.stats import pearsonr
import scipy

# Load the audio files
sample_rate1, audio_data1 = wavfile.read("clap.wav")
sample_rate2, audio_data2 = wavfile.read("tap.wav")

peak1 = np.argmax(audio_data1)
peak2 = np.argmax(audio_data2)


# Compute the spectrograms
_, _, spectrogram_data1 = spectrogram(audio_data1[peak1-1000:peak1+4500], fs=sample_rate1, nperseg=256, noverlap=128)
_, _, spectrogram_data2 = spectrogram(audio_data2[peak2-1000:peak2+4500], fs=sample_rate2, nperseg=256, noverlap=128)

spectrogram_data1 /= np.mean(np.abs(spectrogram_data1))
spectrogram_data2 /= np.mean(np.abs(spectrogram_data2))

# Flatten the spectrogram arrays
spec_flat1 = spectrogram_data1.flatten()
spec_flat2 = spectrogram_data2.flatten()

from user_ml_modules import MAELinearRegression

model = MAELinearRegression()

X = np.expand_dims(spec_flat1, 1)
model.fit(X, spec_flat2)

spectrogram_data1_new = model.predict(X).reshape(spectrogram_data1.shape)


# 1. Calculate correlation factor
correlation_factor, _ = pearsonr(spec_flat1, spec_flat2)

import matplotlib.pyplot as plt

plt.imshow(np.log(np.concatenate([np.abs(spectrogram_data1_new)+0.000001, spectrogram_data2+0.000001], axis=1)))
plt.show()

print("Correlation Factor:", correlation_factor)

# 2. Calculate mean squared error (MSE) between amplitudes
mse = np.mean((spec_flat1 - spec_flat2) ** 2)
print("Mean Squared Error (MSE):", mse)

# 3. Compare frequency content (e.g., using dynamic time warping or cosine similarity)
# (You may need to install additional libraries like fastdtw or scipy.spatial.distance)

# Example of using Dynamic Time Warping (DTW) for frequency comparison
# from fastdtw import fastdtw

# Compute DTW distance
# distance, _ = fastdtw(spec_flat1, spec_flat2)
# print("Dynamic Time Warping (DTW) Distance:", distance)
