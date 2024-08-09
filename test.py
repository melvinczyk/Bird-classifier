import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from collections import Counter
import librosa
import librosa.display

class BirdClassifierCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(BirdClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def save_mel_spectrogram(signal, sr):
    params = {
        'n_fft': 1024,
        'hop_length': 1024,
        'n_mels': 128,
        'win_length': 1024,
        'window': 'hann',
        'htk': True,
        'fmin': 1400,
        'fmax': sr/2
    }

    S = librosa.feature.melspectrogram(y=signal, sr=sr, **params)
    S_dB = librosa.power_to_db(S ** 2, ref=np.max)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), frameon=False)
    ax.set_axis_off()
    librosa.display.specshow(S_dB, sr=sr, fmin=params['fmin'], ax=ax)
    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = Image.fromarray(image)
    plt.close(fig)
    return image

def wav_to_spectrograms(wav_file, segment_length=10):
    waveform, sr = librosa.load(wav_file, sr=16000)
    num_segments = len(waveform) // (sr * segment_length)
    spectrograms = []

    for i in range(num_segments):
        start = i * sr * segment_length
        end = start + sr * segment_length
        segment = waveform[start:end]
        spectrogram = save_mel_spectrogram(segment, sr)
        spectrograms.append(spectrogram)

    return spectrograms

def test_model(wav_file, model_path):
    model = BirdClassifierCNN(num_classes=29)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    spectrograms = wav_to_spectrograms(wav_file)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    predictions = []

    with torch.no_grad():
        for spectrogram in spectrograms:
            image = transform(spectrogram).unsqueeze(0)
            output = model(image)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())


    most_common_prediction = Counter(predictions).most_common(1)[0][0]
    return most_common_prediction

if __name__ == "__main__":
    wav_file = "./clean/TuftedTitmouse/701551.wav"
    model_path = "best_model.pth"
    class_index = test_model(wav_file, model_path)
    print(f"Predicted class index: {class_index}")
