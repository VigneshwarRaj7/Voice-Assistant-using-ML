import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import cv2
# Load images

import pyaudio
import wave

# Parameters for recording
FORMAT = pyaudio.paInt16  # Format of the audio samples
CHANNELS = 1               # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100               # Sample rate (samples per second)
CHUNK = 1024               # Number of frames per buffer

# Create an instance of PyAudio
audio = pyaudio.PyAudio()

# Open a new stream for recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# Record audio for a specified duration (in seconds)
RECORD_SECONDS = 2
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate the PyAudio instance
audio.terminate()

# Save the recorded audio to a WAV file
wf = wave.open("recorded_audio.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print("Audio saved as 'recorded_audio.wav'.")





# rate, data = wavfile.read('trial.wav') # reading wave file.
# print ('All_data =',data)
# print('Number of sample in DATA =',len(data))

# c=data[0:499]             # reading first 500 samples from data variable with contain 200965 samples.
# print('Number of sample in C =',len(c))



# plt.hist(c, bins='auto')  # arguments are passed to np.histogram.
# plt.title("Histogram with 'auto' bins")
# plt.show()
# plt.savefig('histogram1.png')

# rate, data = wavfile.read('recorded_audio.wav') # reading wave file.
# print ('All_data =',data)
# print('Number of sample in DATA =',len(data))

# c=data[0:499]             # reading first 500 samples from data variable with contain 200965 samples.
# print('Number of sample in C =',len(c))



# plt.hist(c, bins='auto')  # arguments are passed to np.histogram.
# plt.title("Histogram with 'auto' bins")
# plt.show()
# plt.savefig('histogram.png')



# image1 = cv2.imread('histogram.jpg')
# image2 = cv2.imread('histogram1.jpg')
# # Calculate histograms
# hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
# hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
# # Compare histograms using correlation
# comparison = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
# print('Correlation:', comparison)