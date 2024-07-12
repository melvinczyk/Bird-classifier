import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

df = pd.read_csv('audio_files.csv')
df.set_index('File', inplace=True)

for f, row in df.iterrows():
    bird_name = row['Bird Name']
    path = f'dataset/audio/{bird_name}/{f}'

    rate, signal = wavfile.read(path)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = df['Bird Name'].unique()
class_dist = df.groupby(['Bird Name'])['length'].mean()


fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f', shadow=False, startangle=90)
ax.axis('equal')
plt.show()
