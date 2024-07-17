import librosa.feature
import os
import noisereduce as nr
import tqdm as tqdm
from scipy.io import wavfile


def reduce_noise(file_path, output_path, sr=44100):
    signal, rate = librosa.load(file_path)
    reduce = nr.reduce_noise(y=signal, sr=rate)
    reduce = (reduce * 32767).astype('int16')
    wavfile.write(output_path, rate, reduce)


def clean_audio(dataset_path):
    if not os.path.exists('clean'):
        os.mkdir('clean')
    elif len(os.listdir('clean')) != 0:
        return

    birds = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for bird in tqdm.tqdm(birds, desc="Processing bird directories", leave=True, position=0):
        bird_path = os.path.join(dataset_path, bird)
        clean_path = os.path.join('clean', bird)

        if not os.path.exists(clean_path):
            os.makedirs(clean_path)
        files = [f for f in os.listdir(bird_path) if os.path.isfile(os.path.join(bird_path, f))]

        for file in tqdm.tqdm(files, desc=f"Cleaning files in {bird}", leave=True, position=0):
            input = os.path.join(bird_path, file)
            output = os.path.join(clean_path, file)
            reduce_noise(input, output)
