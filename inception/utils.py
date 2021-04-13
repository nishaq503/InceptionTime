import os
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODEL_DIR = os.path.join(ROOT_DIR, 'saved_models')
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

DIRS = [
    DATA_DIR,
    LOGS_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
]

DATA_PATHS = [
    (os.path.join(DATA_DIR, 'collection-2.npy'), [3, 4, 1, 2, 1, 1, 0, 0]),
    (os.path.join(DATA_DIR, 'collection-3.npy'), [1, 1, 3, 4, 1, 0, 2, 0]),
    (os.path.join(DATA_DIR, 'collection-4.npy'), [0, 4, 1, 3, 2, 0, 0, 1]),
    (os.path.join(DATA_DIR, 'collection-5.npy'), [3, 0, 0, 1, 1, 2, 4, 1]),
    (os.path.join(DATA_DIR, 'collection-6.npy'), [1, 2, 1, 3, 1, 4, 0, 0]),
]

NUM_FEATURES = 4
NUM_OUTLETS = 8
NUM_CLASSES = 5

TEST_WINDOW_LENGTH = 512

TEST_MODEL_PARAMS = {
    'input_shape': (TEST_WINDOW_LENGTH, NUM_FEATURES),
    'num_classes': NUM_CLASSES,
    'num_modules': 6,
    'bottleneck_size': 32,
    'kernel_size': 40,
    'num_filters': 32,
    'strides': 1,
}

TEST_TRAIN_PARAMS = {
    'num_epochs': 128,
    'verbose': 1,
    'es_schedule': [1e-4, 32],
    'lr_schedule': [0.1, 16, 4],
}

TEST_DATAGEN_PARAMS = {
    'num_classes': NUM_CLASSES,
    'window_length': TEST_WINDOW_LENGTH,
    'batch_size': 16 * NUM_OUTLETS,
    'batches_per_epoch': 8,
}


def create_dirs():
    for _dir in DIRS:
        os.makedirs(_dir, exist_ok=True)
    return


def increment_model_number(n: int):
    filepath = f'build/all_outlets_{n}'
    while os.path.exists(filepath):
        n += 1
        filepath = f'build/all_outlets_{n}'
    return n


def calculate_metrics(
        y_true: np.array,
        y_pred: np.array,
) -> Tuple[float, float, float]:
    precision: float = precision_score(y_true, y_pred, average='macro')
    accuracy: float = accuracy_score(y_true, y_pred)
    recall: float = recall_score(y_true, y_pred, average='macro')
    return precision, accuracy, recall
