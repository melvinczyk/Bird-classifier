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


def saveMel(signal, directory, sr):

    N_FFT = 1024
    HOP_SIZE = 1024
    N_MELS = 128
    WIN_SIZE = 1024
    WINDOW_TYPE = 'hann'
    FEATURE = 'mel'
    FMIN = 1400

    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(6, 6)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    S = librosa.feature.melspectrogram(y=signal, sr=sr,
                                       n_fft=N_FFT,
                                       hop_length=HOP_SIZE,
                                       n_mels=N_MELS,
                                       htk=True,
                                       fmin=FMIN,
                                       fmax=sr / 2)
    librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN)

    fig.savefig(directory)
    plt.ioff()
    fig.clf()
    ax.cla()
    plt.clf()
    plt.close('all')


def save_mel_spectrogram(signal, sr, file_path):
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


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
                        saveMel(signal[start:end], segment_path, sr)

    print('Processing completed.')
process_audio_files(dataset, 'audio_files.csv', mel_path, size)