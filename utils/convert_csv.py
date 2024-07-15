import os
import csv
from scipy.io import wavfile


def generate_csv(dataset_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['File', 'Bird Name', 'Length'])

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                bird_name = os.path.basename(root)
                file_path = os.path.join(root, file)

                rate, signal = wavfile.read(file_path)
                length = signal.shape[0]/rate
                if length <= 1.5:
                    print(f'Removing corrupted file" {file_path}')
                    os.remove(file_path)
                    continue
                csvwriter.writerow([file, bird_name, length])


if __name__ == "__main__":
    dataset = '../dataset/audio'
    output = '../audio_files.csv'

    generate_csv(dataset, output)
