import librosa

from utils import clean
from utils import convert_csv, download
import preprocess
from utils import data_plots

#
size = {
    'desired': 10,
    'minimum': 4,
    'stride': 0,
    'name': 5
}

# dataset = './dataset/audio'
# download.process()
# download.convert_all(dataset)
# convert_csv.generate_csv('./dataset/audio', './audio_files.csv')
# clean.clean_audio('./dataset/audio')
# preprocess.process_audio_files('./clean', './audio_files.csv', './mels', size)

if __name__ == "__main__":
    # download.convert_all('./dataset/audio')
    # convert_csv.generate_csv('./dataset/audio', 'audio_files.csv')
    # data_plots.plot_class_dist('audio_files.csv')
    # clean.clean_audio('dataset/audio')
    # convert_csv.generate_mel_csv('./mels', 'mels_csv.csv')
    # data_plots.plot_mel_class_dist('mels_csv.csv')
    # download.convert_to_wav("test_audio/mourning_dove.mp3")
   #  clean.clean_audio("test_audio")
    signal, sr = librosa.load("./test_audio/mourning_dove.wav")
    preprocess.save_mel_spectrogram(signal, "./test_audio", sr)
