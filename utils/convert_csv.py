import os
import csv


def generate_csv(dataset_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['File', 'Bird Name'])

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                bird_name = os.path.basename(root)
                file_path = os.path.join(root, file)
                csvwriter.writerow([file, bird_name])


if __name__ == "__main__":
    dataset = '../dataset/audio'
    output = '../audio_files.csv'

    generate_csv(dataset, output)
