import os
import pandas as pd
from pydub import AudioSegment
import numpy as np
import librosa

df = pd.read_csv('audio_files.csv')

for index, row in df.iterrows():
    bird_name = row['Bird Name'].strip()
    file_name = row['File'].strip()
    file_path = f'dataset/audio/{bird_name}/{file_name}'

    signal, sample_rate = librosa.load(file_path, sr=None)
    print(f"Processed file: {file_path}")