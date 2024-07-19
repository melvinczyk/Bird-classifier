import os
import pandas as pd

train = 0.8
val = 0.1
test = 0.1
kfolds = 1

mels_path = './mels/'
split_path = './mels/split/'

birds = []
bird_data = []
all_files = []

for root ,dirs, files in os.walk(mels_path):
    for dir in dirs:
        birds.append(dir)

print(birds)

train_set = []
test_set = []
val_set = []

df = pd.read_csv('audio_files.csv')
for nr, bird in enumerate(birds):
    for root, dirs, files in os.walk(mels_path + bird):
        continue