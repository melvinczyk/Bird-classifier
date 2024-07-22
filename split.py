import os
import pandas as pd


train = 0.8
val = 0.1
test = 0.1

birds = []
all_wav_files = []

df = pd.read_csv('audio_files.csv')

for index, row in df.iterrows():
    birds.append(row['Bird Name'])
    all_wav_files.append([row['Bird Name'], row["File"]])
birds = list(set(birds))
print(birds)
print(all_wav_files)
