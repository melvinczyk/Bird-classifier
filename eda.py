import os
import pandas as pd
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tqdm
from python_speech_features import mfcc, logfbank


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

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

def plot_fbank(fbank):
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

df = pd.read_csv('audio_files.csv')
df.set_index('File', inplace=True)

for f, row in df.iterrows():
    bird_name = row['Bird Name'].strip()
    path = f'dataset/audio/{bird_name}/{f}'

    rate, signal = wavfile.read(path)
    df.at[f, 'length'] = signal.shape[0]/rate

df.to_csv('audio_files.csv')
birds = df[df.columns[0]].to_numpy()
classes = df['Bird Name'].unique()
class_dist = df.groupby(['Bird Name'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class dist', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f')
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    row = df[df['Bird Name'] == c].iloc[0]
    file = row['File']
    bird = row['Bird Name'].strip()

    path = os.path.join('dataset/audio', bird, file)

    try:
        signal, rate = librosa.load(path, sr=44100)
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals[c] = signal
        fft[c] = calc_fft(signal, rate)

        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        fbank[c] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel
        print(f"Processed file: {path}")
    except Exception as e:
        print(f"Error in processing {path}: {e}")

plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

if len(os.listdir('clean')) == 0:
    for f in tqdm.tqdm(df['File']):
        path = os.path.join('dataset/audio', row['Bird Name'].strip(), row['File'])
        signal, rate = librosa.load(path, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])
    
