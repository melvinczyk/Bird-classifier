import os
from tqdm import tqdm
import pandas as pd
import librosa
from scipy.io import wavfile

if len(os.listdir('clean')) == 0:
    for f in tqdm.tqdm(df['File']):
        path = os.path.join('dataset/audio', df['Bird Name'], f)
        signal, rate = librosa.load(path, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])