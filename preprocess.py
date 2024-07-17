import matplotlib.pyplot as plt
import os
import librosa
import numpy as np
import librosa.feature
import tqdm
import pandas as pd


dataset = './clean'
mel_path = './mels'

size = {
    'desired': 10,
    'minimum': 4,
    'stride': 0,
    'name': 5
}


def save_mel_spectrogram(signal, directory, sr):
    params = {
        'n_fft': 1024,
        'hop_length': 1024,
        'n_mels': 128,
        'win_length': 1024,
        'window': 'hann',
        'htk': True,
        'fmin': 1400,
        'fmax': sr/2
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), frameon=False)
    ax.set_axis_off()

    S = librosa.feature.melspectrogram(y=signal, sr=sr, **params)
    S_dB = librosa.power_to_db(S ** 2, ref=np.max)
    librosa.display.specshow(S_dB, fmin=params['fmin'], ax=ax)
    fig.savefig(directory)
    plt.close(fig)


def process_audio_files(dataset_path, csv_path, mels_path, size):
    df = pd.read_csv(csv_path)
    print('Number of directories to check and cut: ', len(df['Bird Name'].unique()))

    step = (size['desired'] - size['stride']) * 16000
    if step <= 0:
        print("Error: Stride should be lower than desired length.")
        return

    for bird_name in df['Bird Name'].unique():
        bird_df = df[df['Bird Name'] == bird_name]
        print("Processing bird: ", bird_name)
        for _, row in tqdm.tqdm(bird_df.iterrows(), total=bird_df.shape[0], desc=f"Processing {bird_name}"):
            file_path = os.path.join(dataset_path, bird_name, row['File'])
            directory = os.path.join(mels_path, bird_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            mel_base = row['File'].rsplit('/',1)[-1].replace(' ', '')[:-4]
            mel_path = os.path.join(directory, f"{mel_base}_{row.name}.png")
            if not os.path.exists(mel_path):
                signal, sr = librosa.load(file_path, sr=16000)
                step = (size['desired'] - size['stride']) * sr

                nr = 0
                for start, end in zip(range(0, len(signal), step), range(size['desired'] * sr, len(signal), step)):
                    nr += 1
                    if end - start > size['minimum'] * sr:
                        segment_path = os.path.join(directory, f"{mel_base}_{nr}.png")
                        save_mel_spectrogram(signal[start:end], segment_path, sr)

    print('Processing completed.')