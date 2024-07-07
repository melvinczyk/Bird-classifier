import sys
import os
from utils.download import *
import platform
import subprocess

def install_ffmpeg():
    if platform.system() == 'Darwin':
        if subprocess.call(['brew', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) != 0:
            print("Homebrew is not installed. Please install Homebrew from https://brew.sh/")
            sys.exit(1)
        print('Installing ffmpeg...')
        subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
    elif platform.system() == 'Linux':
        subprocess.run(['apt-get', 'install', 'ffmpeg'], check=True)
    elif platform.system() == 'Windows':
        subprocess.run(['choco', 'install', 'ffmpeg'], check=True)

def main():
    install_ffmpeg()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

if __name__ == '__main__':
    main()
    if not os.path.exists('dataset/audio'):
        process()
    convert_all('dataset/audio')