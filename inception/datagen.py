import os
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(
            self,
            data_paths: List[Tuple[str, List[int]]],
            num_classes: int,
            window_length: int,
            batch_size: int,
            batches_per_epoch: int,
    ):
        """ This class batches data for the model.
        :param data_paths: A list of tuples where each tuple is ('/path/to/dataset.npy', list of integer labels).
                           The labels are expected as a list of integers for devices in each outlet.
                               - 0 -> outlet is off or nothing plugged in or device is off.
                               - 1 -> good device
                               - 2 -> fault mode 1
                               - 3 -> fault mode 2
                               - 4 -> fault mode 4
        :param num_classes: Number of classes for prediction.
        :param window_length: number of contiguous rows to pull for each point.
        :param batch_size: number of windows to pull in each batch. This should be:
                                - a multiple of num_outlets, and
                                - and be at least 32 * num_outlets.
        :param batches_per_epoch: number of batches for each epoch.
        """
        self.num_outlets: int = len(data_paths[0][1])
        self.num_classes: int = num_classes
        label_set: Set[int] = set(range(self.num_classes))

        for filepath, labels in data_paths:
            if not os.path.exists(filepath):
                raise ValueError(f'Dataset {filepath} does not exist. Please verify that the dataset is present in {utils.DATA_DIR}')

            if len(labels) != self.num_outlets:
                raise ValueError(f'Expected the same number of labels for each data file. '
                                 f'Got {self.num_outlets} for the first file and {len(labels)} for {filepath}.')

            if not all((isinstance(label, int) for label in labels)):
                raise ValueError(f'Labels must all be integers. Got {labels} for {filepath}.')

            if not set(labels) <= label_set:
                raise ValueError(f'Labels must be integers from 0 to {self.num_outlets}. Got {labels} for {filepath}.')

        if batch_size % self.num_outlets != 0:
            raise ValueError(f'batch_size must be a multiple of {self.num_outlets}. Got {batch_size} instead.')

        self.datasets: List[Tuple[np.array, List[int]]] = list()
        for filepath, labels in data_paths:
            data: np.array = np.load(filepath)
            if window_length < data.shape[0]:
                self.datasets.append((data, labels))
            else:
                raise ValueError(f'window_length should be smaller than the number of rows in the {filepath} dataset. '
                                 f'Got num_rows: {data.shape[0]}, window_length: {window_length}')

        self.window_length: int = window_length
        self.batch_size: int = batch_size
        self.batches_per_epoch: int = batches_per_epoch

        self.on_epoch_end()

    def __len__(self) -> int:
        """ Returns the number of batches per epoch. """
        return self.batches_per_epoch

    def __getitem__(self, index):
        """ Returns one batch of data. """
        return self.data_generation()

    def on_epoch_end(self):
        """ Wrap up an epoch. """
        # TODO: Some sort of shuffling?
        return

    def data_generation(self):
        """ Generate one batch of data for the neural network. """
        x: np.array = np.zeros(
            shape=(self.batch_size, self.window_length, 4),
            dtype=np.float32,
        )
        y: np.array = np.zeros(shape=(self.batch_size, ), dtype=np.uint8)

        for i in range(0, self.batch_size, self.num_outlets):
            # use the j-th dataset and labels
            j: int = np.random.randint(0, len(self.datasets)) if len(self.datasets) > 1 else 0
            data, labels = self.datasets[j]

            for k, s in enumerate(np.random.permutation(self.num_outlets)):
                columns = [
                    1,                             # VRMS
                    s + 2,                         # IRMS
                    s + 2 + self.num_outlets,      # WATTHR
                    s + 2 + 2 * self.num_outlets,  # VARHR
                ]
                # randomly sample a starting row
                start_row = np.random.randint(low=0, high=data.shape[0] - self.window_length)
                end_row = start_row + self.window_length

                # pull a window starting at that row
                indices = list(range(start_row, end_row))
                point = data[indices][:, columns]  # take only the relevant columns

                # insert point and label into the x and y arrays
                x[i + k] = point
                y[i + k] = labels[s]

        return x, y


if __name__ == '__main__':
    import utils

    _datagen = DataGenerator(utils.RAW_DATA_PATHS, **utils.TEST_DATAGEN_PARAMS)
    _x, _y = _datagen.data_generation()
    print(f'batch_shape: {_x.shape}')
    print(f'batch_size: {len(_y)}')
    for _i in range(0, _datagen.batch_size, 8):
        line = ', '.join(map(str, _y[_i: _i + 8]))
        if ((_i // 8) % 4) == 3:
            print(f' {line},')
        else:
            print(f' {line},', end=' ')
