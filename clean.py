import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import gc
import os
import numpy as np
import warnings
import tqdm as tqdm
warnings.filterwarnings("ignore")
from scipy.io import wavfile


def saveMel(signal, dir):
    gc.enable()
    N_FFT = 1024
    HOP_SIZE = 1024
    N_MELS = 128
    WIN_SIZE = 1024
    WINDOW_TYPE = 'hann'
    FEATURE_TYPE = 'mel'
    FMIN = 1400

    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(6,6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    S = librosa.feature.melspectrogram(y=signal, sr=16000,
                                       n_fft=N_FFT,
                                       hop_length=HOP_SIZE,
                                       n_mels=N_MELS,
                                       htk=True,
                                       fmin=FMIN,  # higher limit ##high-pass filter freq.
                                       fmax=16000 / 2)  # AMPLITUDE
    librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN)  # power = S**2

    fig.savefig(dir)
    plt.ioff()
    # plt.show(block=False)
    fig.clf()
    ax.cla()
    plt.clf()
    plt.close('all')


def clean():
    for root, dirs, files in os.walk('clean'):
        for file in files:
            path = os.path.join(root, file)
            rate, signal = wavfile.read(path)
            mel = './mels' + '/' + file+".png"
            saveMel(signal, mel)

# waveform
file = 'dataset/audio/AmericanCrow/12786.wav'
signal, sr = librosa.load(file)
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft)

freq = np.linspace(0, sr, len(magnitude))

left_freq = freq[:len(freq) // 2]
left_magnitude = magnitude[:len(freq) // 2]
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude [dB]")
plt.plot(left_freq, left_magnitude)
plt.show()

# stft -> spectrogram

n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency [Hz]")
plt.colorbar()
plt.show()
