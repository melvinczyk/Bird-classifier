import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


y, sr = librosa.load('dataset/audio/AmericanCrow/12786.wav')
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Monophonic - file number 1')
plt.show()

plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram - file number 1')
plt.show()

