# Run this to get dataset
import subprocess

def process():
    with open('popular_birds.txt', 'r') as file:
        lines = file.readlines()

    for line in lines:
        entry = line.strip()
        if entry:
            command = f'xeno-canto -dl {entry} q:A cnt:United_States'
            subprocess.run(command, shell=True)


if __name__ == '__main__':
    process()
