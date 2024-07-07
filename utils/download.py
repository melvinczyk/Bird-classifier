# Run this to get dataset
import subprocess
from pydub import AudioSegment
import os

def process():
    with open('popular_birds.txt', 'r') as file:
        lines = file.readlines()

    for line in lines:
        entry = line.strip()
        if entry:
            command = f'xeno-canto -dl {entry} q:A cnt:United_States'
            subprocess.run(command, shell=True)

def convert_to_wav(path):
    try:
        audio = AudioSegment.from_mp3(path)
        wav_file = path.replace(".mp3", ".wav")
        audio.export(wav_file, format='wav')
        os.remove(path)
        print(f"Converted file {path} to {wav_file}")
    except Exception as e:
        print(f"Error convertig {path}: {e}")

def convert_all(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".mp3"):
                path = os.path.join(root, file)
                convert_to_wav(path)

if __name__ == '__main__':
    dataset = 'dataset/audio'
    if not os.path.exists(dataset):
        process()
    convert_all(dataset)
