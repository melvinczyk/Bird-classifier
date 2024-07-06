import sys
import platform
import subprocess
from setuptools import setup, find_packages

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
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    install_ffmpeg()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

if __name__ == '__main__':
    main()