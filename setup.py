import sys
from utils.download import *
import platform
import subprocess
from utils.convert_csv import generate_csv


def install_ffmpeg():
    if platform.system() == 'Darwin':
        if subprocess.call(['brew', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) != 0:
            print("Homebrew is not installed. Please install Homebrew from https://brew.sh/")
            sys.exit(1)
        print('Installing ffmpeg...')
        subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
    elif platform.system() == 'Linux':
        subprocess.run(['apt-get', 'install', 'ffmpeg'], check=True)


if __name__ == '__main__':
    install_ffmpeg()
    if not os.path.exists('dataset/audio'):
        process()
    convert_all('dataset/audio')
    generate_csv('dataset/audio', 'audio_files.csv')
