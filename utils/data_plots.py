import matplotlib.pyplot as plt
import librosa
import pandas as pd
import numpy as np


def plot_signals(signal_paths):
    fig, ax = plt.subplots(sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    y_offset = 0
    for path in signal_paths:
        signal, sr = librosa.load(path, sr=16000)
        times = np.arange(len(signal)) / sr
        ax.plot(times, signal + y_offset, label=f"Signal {len(ax.lines) +1}")
        y_offset += np.max(np.abs(signal)) * 2

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Amplitude')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_spectrogram(file_path, n_fft=2048, hop_length=1024):
    signal, sr = librosa.load(file_path, sr=16000)

    stft = librosa.core.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.title(f'{file_path} Spectrogram')
    plt.ylabel("Frequency [Hz]")
    plt.colorbar()
    plt.show()


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('FFT', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


def plot_class_dist(files_csv):
    df = pd.read_csv(files_csv)
    df.set_index('File', inplace=True)
    birds = df[df.columns[0]].to_numpy()
    classes = df['Bird Name'].unique()

    class_dist = df.groupby('Bird Name')['Length'].sum()
    class_dist = class_dist.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title('Class Distribution')
    class_dist.plot(kind='barh', ax=ax)
    ax.set_xlabel('Total Length (seconds)')
    ax.set_ylabel('Bird Name')
    plt.show()


def plot_fbank(fbank, mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coeff', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1