import os
import pandas as pd
import numpy as np
import scipy.io as wavfile
import librosa

df = pd.read_csv('audio_files.csv')

for index, row in df.iterrows():
    bird_name = row['Bird Name']
    file_name = row['File']
    file_path = f'dataset/{bird_name}/{file_name}'
    rate, signal = wavfile.hb_read(file_path)
    print(f"Processed file: {file_path}")