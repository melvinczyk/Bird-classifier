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


df = pd.read_csv('audio_files.csv')
df.set_index('File', inplace=True)

for f, row in df.iterrows():
    bird_name = row['Bird Name'].strip()
    path = f'dataset/audio/{bird_name}/{f}'

    rate, signal = wavfile.read(path)
    length = signal.shape[0]/rate
    if length <= 1.5:
        print(f'Removing corrupted file: {path}')
        os.remove(path)
        continue
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

if len(os.listdir('clean')) == 0:
    for root, dirs, files in os.walk('dataset/audio'):
        for f in tqdm.tqdm(files):
            path = os.path.join(root, f)
            signal, rate = librosa.load(path, sr=16000)
            mask = envelope(signal, rate, 0.0005)
            clean_path = os.path.join('clean', f)
            wavfile.write(clean_path, rate, signal[mask])
