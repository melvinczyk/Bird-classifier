from utils import clean
from utils import convert_csv, download
import preprocess

#
# size = {
#     'desired': 10,
#     'minimum': 4,
#     'stride': 0,
#     'name': 5
# }
#
# dataset = './dataset/audio'
# download.process()
# download.convert_all(dataset)
# convert_csv.generate_csv('./dataset/audio', './audio_files.csv')
# clean.clean_audio('./dataset/audio')
# preprocess.process_audio_files('./clean', './audio_files.csv', './mels', size)

if __name__ == "__main__":
    convert_csv.generate_csv('./clean', './audio_files.csv')