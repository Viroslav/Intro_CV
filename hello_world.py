import numpy as np


def predict(img: np.ndarray, model_path: str):
    with open(model_path, 'rb') as f:
        a = np.load(f)
        return (img - a[0])/a[1]


def train(img: np.ndarray, save_model_path: str):
    model = np.array([img.mean(), img.std()])
    return np.save(save_model_path, model)
