import matplotlib.pyplot as plt
import os
import librosa
import numpy as np
import librosa.feature
import tqdm
import pandas as pd


dataset = './clean'
mel_path = './mels_5_sec'

size = {
    'desired': 5,
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


def process_audio_files(dataset_path, mels_path, size):
    bird_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print('Number of directories to check and cut: ', len(bird_names))

    step = (size['desired'] - size['stride']) * 16000
    if step <= 0:
        print("Error: Stride should be lower than desired length.")
        return

    for bird_name in bird_names:
        bird_folder = os.path.join(dataset_path, bird_name)
        print("Processing bird: ", bird_name)

        output_directory = os.path.join(mels_path, bird_name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for file_name in tqdm.tqdm(os.listdir(bird_folder), desc=f"Processing {bird_name}"):
            file_path = os.path.join(bird_folder, file_name)
            if os.path.isfile(file_path):
                mel_base = file_name.rsplit('.', 1)[0].replace(' ', '')
                mel_path = os.path.join(output_directory, f"{mel_base}.png")

                if not os.path.exists(mel_path):
                    signal, sr = librosa.load(file_path, sr=16000)
                    step = (size['desired'] - size['stride']) * sr

                    nr = 0
                    for start, end in zip(range(0, len(signal), step), range(size['desired'] * sr, len(signal), step)):
                        nr += 1
                        if end - start > size['minimum'] * sr:
                            segment_path = os.path.join(output_directory, f"{mel_base}_{nr}.png")
                            if os.path.exists(segment_path):
                                print(f"{segment_path} exists, skipping")
                                continue
                            save_mel_spectrogram(signal[start:end], segment_path, sr)


    print('Processing completed.')


def process_audio_folder(dataset_path, mels_path, size):
    if not os.path.exists(mels_path):
        os.makedirs(mels_path)

    audio_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    print('Number of files to process: ', len(audio_files))

    step = (size['desired'] - size['stride']) * 16000
    if step <= 0:
        print("Error: Stride should be lower than desired length.")
        return

    for file_name in tqdm.tqdm(audio_files, desc="Processing files"):
        file_path = os.path.join(dataset_path, file_name)
        if os.path.isfile(file_path):
            mel_base = file_name.rsplit('.', 1)[0].replace(' ', '')
            mel_path = os.path.join(mels_path, f"{mel_base}.png")

            if not os.path.exists(mel_path):
                signal, sr = librosa.load(file_path, sr=16000)
                step = (size['desired'] - size['stride']) * sr

                nr = 0
                for start, end in zip(range(0, len(signal), step), range(size['desired'] * sr, len(signal), step)):
                    nr += 1
                    if end - start > size['minimum'] * sr:
                        segment_path = os.path.join(mels_path, f"{mel_base}_{nr}.png")
                        if os.path.exists(segment_path):
                            print(f"{segment_path} exists, skipping")
                            continue
                        save_mel_spectrogram(signal[start:end], segment_path, sr)

    print('Processing completed.')