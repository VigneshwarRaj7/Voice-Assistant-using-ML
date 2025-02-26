import librosa
import numpy as np

def get_mfcc_features(file_path):
    # Load an audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    audio_data = audio_data.astype(np.float64)

    # Find the peak and extract chunks around the peak
    peak = np.argmax(np.abs(audio_data))
    x_chunk = audio_data[peak-512:peak+4096-512]

    # Normalize according to the peak neighbor intervals
    x_xchunk = audio_data[peak-16:peak+16]
    energy = np.mean(np.abs(x_xchunk))
    audio_data_normalized = x_chunk / energy

    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data_normalized, sr=sample_rate, n_mfcc=13, hop_length=512, n_fft=2048)
    mfccs_flat = mfccs.flatten()
    
    return mfccs_flat