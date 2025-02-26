import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
import os
# Load the audio file and compute its spectrogram
def compute_spectrogram(audio_file):
    sample_rate, audio_data = wavfile.read(audio_file)
    _, _, spectrogram_data = spectrogram(audio_data, fs=sample_rate)
    return spectrogram_data

# Define the RNN model
def build_rnn_model(input_shape):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Load and preprocess the spectrogram data
# spectrogram_data1 = compute_spectrogram("clap1.wav")
# spectrogram_data2 = compute_spectrogram("nothing.wav")
# spectrogram_data3 = compute_spectrogram("clap2.wav")
# Assuming the spectrogram data has the same shape, you may need to adjust this if it's not the case

sample_rate1, audio_data1 = wavfile.read("/Users/fiery_stallion/Downloads/college/engingeering project/codes/sound-detection-Vigneshwar Raj/clap.wav")
sample_rate2, audio_data2 = wavfile.read("/Users/fiery_stallion/Downloads/college/engingeering project/codes/sound-detection-Vigneshwar Raj/tap.wav")

peak1 = np.argmax(audio_data1)
peak2 = np.argmax(audio_data2)

_, _, spectrogram_data1 = spectrogram(audio_data1[peak1-1000:peak1+4500], fs=sample_rate1, nperseg=256, noverlap=128)
_, _, spectrogram_data2 = spectrogram(audio_data2[peak2-1000:peak2+4500], fs=sample_rate2, nperseg=256, noverlap=128)

spectrogram_data1 /= np.mean(np.abs(spectrogram_data1))
spectrogram_data2 /= np.mean(np.abs(spectrogram_data2))

# Flatten the spectrogram arrays
# spec_flat1 = spectrogram_data1.flatten()
# spec_flat2 = spectrogram_data2.flatten()

data_rows = []
labels= []
folder_paths = ['clap_sounds', 'tap_sounds', 'no_sounds']
for label, folder_path in enumerate(folder_paths):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            # audio_data, sample_rate = librosa.load(file_path, sr=None)
            # sample_rate, audio_data = wavfile.read(file_path)
            # peak = np.argmax(audio_data)
            # _, _, spectrogram_data = spectrogram(audio_data[peak-1000:peak1+4500], fs=sample_rate, nperseg=256, noverlap=128)
            # spectrogram_data /= np.mean(np.abs(spectrogram_data))

            spectrogram_data = compute_spectrogram(file_path)
            data_rows.append(spectrogram_data)
            if folder_path  == 'clap_sounds':
                labels.append(0)
            elif folder_path == 'tap_sounds':
                labels.append(1)
            else:
                labels.append(2)



spectrogram_data = compute_spectrogram("/Users/fiery_stallion/Downloads/college/engingeering project/codes/sound-detection-Vigneshwar Raj/clap.wav")
sd = []
sd.append(spectrogram_data)



print(labels)
input_shape = data_rows[1].shape

# Prepare data for RNN
X = np.stack(data_rows)
y = np.stack(labels)  # Example binary labels, adjust as needed
test = np.stack(sd)
test_labels = (0.0).dtype(float)
# # Build the RNN model
model = build_rnn_model(input_shape)
data_rows
# # Train the model
model.fit(X, y, epochs=10, batch_size=32)
y_pred = model.predict(test)
loss, accuracy = model.evaluate(X, y)
print("Model accuracy:", accuracy)
# print("Test Accuracy: {:.2f}%".format(accuracy_score(y_pred, test_labels)*100))
# # Evaluate the model
# loss, accuracy = model.evaluate(X, y)
# print("Model loss:", loss)
# print("Model accuracy:", accuracy)
