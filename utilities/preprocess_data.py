# utilities/preprocess_data.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Function to preprocess FER2013 dataset
def preprocess_fer2013(input_csv, output_dir, emotion_labels):
    os.makedirs(output_dir, exist_ok=True)
    for usage in ['train', 'val', 'test']:
        for label in emotion_labels.values():
            os.makedirs(os.path.join(output_dir, usage, label), exist_ok=True)

    df = pd.read_csv(input_csv)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing images'):
        pixels = np.array(row['pixels'].split(), dtype='uint8')
        image = pixels.reshape(48, 48)
        img = Image.fromarray(image)
        label = emotion_labels[row['emotion']]
        usage = row['Usage']
        if usage == 'Training':
            path = os.path.join(output_dir, 'train', label)
        elif usage == 'PublicTest':
            path = os.path.join(output_dir, 'val', label)
        else:
            path = os.path.join(output_dir, 'test', label)
        img.save(os.path.join(path, f'{index}.jpg'))

if __name__ == "__main__":
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    preprocess_fer2013('fer2013/fer2013.csv', 'FER2013_processed', emotion_labels)