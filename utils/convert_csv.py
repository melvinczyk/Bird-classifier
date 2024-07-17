import os
import csv
import pandas as pd
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


def formalize_csv(dataset_path, csv_path):
    if not os.path.exists(csv_path):
        generate_csv(dataset_path, csv_path)
    df = pd.read_csv(csv_path)
    valid_files = set((row['Bird Name'], row['File']) for _, row in df.iterrows())
    for root, dirs, files in os.walk(dataset_path):
        bird_name = os.path.basename(root)
        for file in files:
            file_path = os.path.join(root, file)
            if (bird_name, file) not in valid_files:
                os.remove(file_path)
                print(f"Removing extra file {file_path}")




if __name__ == "__main__":
    dataset = '../dataset/audio'
    output = '../audio_files.csv'

    generate_csv(dataset, output)
