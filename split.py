import os
import pandas as pd


train = 0.8
val = 0.1
test = 0.1

train_set = []
val_set = []
test_set = []

df = pd.read_csv('audio_files.csv')

bird_files = {}

for index, row in df.iterrows():
    bird_name = row['Bird Name']
    file_name = row['File']

    if bird_name in bird_files:
        bird_files[bird_name].append(file_name)
    else:
        bird_files[bird_name] = [file_name]

for bird, files in bird_files.items():
    num_files = len(bird_files.get(bird, []))

    train_set.append(int(train * num_files))
    val_set.append(int(val * num_files))
    test_set.append(int(test * num_files) + num_files - (int(train * num_files) + int(val * num_files) + int(test * num_files)))
    print(f"Found {len(bird_files.get(bird, []))} files for {bird}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")


