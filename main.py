import librosa

from utils import clean
from utils import convert_csv, download
import preprocess
from utils import data_plots
import requests
import os


size = {
    'desired': 5,
    'minimum': 3,
    'stride': 0,
    'name': 5
}

# dataset = './dataset/audio'
# download.process()
# download.convert_all(dataset)
# convert_csv.generate_csv('./dataset/audio', './audio_files.csv')
# clean.clean_audio('./dataset/audio')
# preprocess.process_audio_files('./clean', './audio_files.csv', './mels', size)

def download_audio_file(asset_num, bird_name):
    url = f"https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{asset_num}"
    output_path = f"./test_audio/{bird_name}/{asset_num}_cornell.wav"
    if not os.path.exists(f"./test_audio/{bird_name}"):
        os.makedirs(f"./test_audio/{bird_name}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"File downloaded successfully and saved to {output_path}")
        clean.reduce_noise(output_path, output_path)
        print(f"Cleaned to path: ./clean/{bird_name}/{asset_num}_cornell.wav")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

if __name__ == "__main__":
    # preprocess.process_audio_files('./clean', './audio_files.csv', './mels_5_sec', size)
    # download.convert_all('./dataset/audio')
    # convert_csv.generate_csv('./dataset/audio', 'audio_files.csv')
    # data_plots.plot_class_dist('audio_files.csv')
    # clean.clean_audio('dataset/audio')
    # convert_csv.generate_mel_csv('./mels_5_sec', 'mels_5_sec_csv.csv')
    num = "580412861"
    bird = "MourningDove"
    #download_audio_file(num, bird)
    #preprocess.process_audio_folder('./clean/MourningDove', './mels_5_sec/MourningDove', size)
    # preprocess.process_audio_files('./clean', 'mels_5_sec', size)
    data_plots.plot_mel_class_dist('./mels_5_sec')
    # download.convert_to_wav("test_audio/mourning_dove.mp3")
    # clean.clean_audio("test_audio")
    # signal, sr = librosa.load("./test_audio/mourning_dove.wav")
    # preprocess.save_mel_spectrogram(signal, "./test_audio", sr)
