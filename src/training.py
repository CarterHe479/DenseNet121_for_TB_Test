import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.data_preparation import prepare_data
from src.densenet_model import densenet

def train_model(train_images_paths, train_labels, model_path=None):
    if model_path:
        model = load_model(model_path)
    else:
        input_shape = (64, 64, 1)
        n_classes = 1
        model = densenet(input_shape, n_classes)
    
    images, labels = prepare_data(train_images_paths, train_labels)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(images, labels, epochs=32, batch_size=50)
    
    return model, history

def save_model(model, model_path, weights_path):
    model.save(model_path)
    model.save_weights(weights_path)
