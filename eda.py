import os
import pandas as pd
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa

df = pd.read_csv('audio_files.csv')
df.set_index('File', inplace=True)

for f, row in df.iterrows():
    bird_name = row['Bird Name'].strip()
    path = f'dataset/audio/{bird_name}/{f}'

    rate, signal = wavfile.read(path)
    df.at[f, 'length'] = signal.shape[0]/rate
    #print(f'Processed file: {path}')

df.to_csv('length.csv')
class_dist = df.groupby(['Bird Name'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class disrt', y=1.08)
ax.pie(class_dist, labels=class_dist.index, auttopct='%1.1f%')
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)