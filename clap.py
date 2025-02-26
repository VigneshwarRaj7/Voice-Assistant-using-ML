import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Load the audio file
sample_rate, audio_data = wavfile.read("recorded_audio.wav")

# Convert stereo audio to mono by taking the mean of the channels
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

# Normalize the audio data
audio_data = audio_data / np.max(np.abs(audio_data))

# Create the spectrogram
frequencies, times, spectrogram_data = spectrogram(audio_data, fs=sample_rate)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data), shading='gouraud')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Spectrogram of Audio Signal')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()
