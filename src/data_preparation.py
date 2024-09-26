import numpy as np
import pandas as pd
from src.image_preprocessing import load_image

def prepare_data(image_paths, prediction_values):
    images = []
    labels = []

    for img_path, label in zip(image_paths, prediction_values):
        img_array = load_image(img_path)
        images.append(img_array)
        labels.append(label)
    
    return np.array(images), np.array(labels)
