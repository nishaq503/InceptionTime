import os
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from tensorflow import keras

from . import utils


class InceptionTime:
    """ My implementation of the InceptionTime network for a Kaggle competition.

    Fawaz, Hassan Ismail, et al. "Inceptiontime: Finding alexnet for time series classification." Data Mining and Knowledge Discovery 34.6 (2020): 1936-1962.
    """

    def __init__(
            self,
            model_name: str,
            input_shape: Tuple,
            num_classes: int,
            num_modules: int = 6,
            bottleneck_size: int = 32,
            kernel_size: int = 40,
            num_filters: int = 32,
            strides: int = 1,
    ):
        """ Builds an InceptionTime Network ready for training.

        :param model_name:
        :param input_shape:
        :param num_classes:
        :param num_modules:
        :param bottleneck_size:
        :param kernel_size:
        :param num_filters:
        :param strides:
        """

        self.model_name: str = model_name
        self.bottleneck_size: int = bottleneck_size
        self.kernel_sizes: List[int] = [kernel_size // (2 ** i) for i in range(3)]
        self.num_filters: int = num_filters
        self.strides: int = strides

        input_layer = keras.layers.Input(input_shape)
        inner_layer, residual = input_layer, input_layer

        for i in range(num_modules):
            inner_layer = self._inception_module(input_layer)
            if i % 3 == 2:
                inner_layer = self._shortcut_layer(residual, inner_layer)
                residual = inner_layer

        pool_layer = keras.layers.GlobalAveragePooling1D()(inner_layer)
        output_layer = keras.layers.Dense(
            units=num_classes,
            activation='softmax',
        )(pool_layer)

        self.model: keras.models.Model = keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
        )

    def compile(
            self,
            optimizer: str = 'adam',
            loss: str = 'sparse_categorical_crossentropy',
    ):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'],
        )
        return

    def summary(self):
        return self.model.summary()

    def draw(self):
        path = os.path.join(utils.PLOTS_DIR, f'{self.model_name}.png')
        keras.utils.plot_model(self.model, path, show_shapes=True)
        return

    def train(
            self,
            train_generator: keras.utils.Sequence,
            validation_generator: keras.utils.Sequence,
            num_epochs: int = 128,
            verbose: int = 1,
            es_schedule: Tuple[float, int] = None,
            lr_schedule: Tuple[float, int, int] = None,
    ):
        callbacks = [keras.callbacks.TensorBoard(
            log_dir=os.path.join(utils.LOGS_DIR, self.model_name),
        )]

        if es_schedule is not None:
            delta, patience = es_schedule
            if patience > num_epochs:
                raise ValueError(f'es_schedule[0], i.e. the number of epochs to wait, should not be greater than the total number of epochs')
            if not isinstance(delta, float) or not (0. < delta):
                raise ValueError(f'es_schedule[1], i.e. the minimum delta by which LR must fall, should be a small, positive float e.g. 1e-4.')
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=delta,
                patience=patience,
                verbose=verbose,
                restore_best_weights=True,
            ))

        if lr_schedule is not None:
            factor, patience, cooldown = lr_schedule
            if not isinstance(factor, float) or not (0. < factor < 1.):
                raise ValueError(f'lr_schedule[0], i.e. the factor by which to reduce the learning rate, should be a float in the (0, 1) range.')
            if not isinstance(patience, int) or patience > num_epochs:
                raise ValueError(f'lr_schedule[1], i.e. the number of epochs to wait for val_loss to fall, should be an int no larger than num_epochs.')
            if not isinstance(cooldown, int) or cooldown > patience:
                raise ValueError(f'lr_schedule[2], i.e. the number of epochs to wait between reducing the LR, '
                                 f'should be an int no larger than lr_patience, i.e. lr_schedule[1].')

        self.model.fit(
            x=train_generator,
            epochs=num_epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_generator,
        )
        return

    def _inception_module(
            self,
            input_tensor,
            activation: str = 'linear',
    ):
        if int(input_tensor.shape[-1]) > 1:
            input_tensor = keras.layers.Conv1D(
                filters=self.num_filters,
                kernel_size=1,
                padding='same',
                activation=activation,
                use_bias=False,
            )(input_tensor)

        conv_layers = [
            keras.layers.Conv1D(
                filters=self.num_filters,
                kernel_size=kernel_size,
                strides=self.strides,
                padding='same',
                activation=activation,
                use_bias=False
            )(input_tensor)
            for kernel_size in self.kernel_sizes
        ]

        pool_layer = keras.layers.MaxPool1D(
            pool_size=3,
            strides=self.strides,
            padding='same',
        )(input_tensor)

        conv_layers.append(keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=1,
            padding='same',
            activation=activation,
            use_bias=False
        )(pool_layer))

        output_tensor = keras.layers.Concatenate(axis=2)(conv_layers)
        output_tensor = keras.layers.BatchNormalization()(output_tensor)
        return keras.layers.Activation(activation='relu')(output_tensor)

    # noinspection PyMethodMayBeStatic
    def _shortcut_layer(
            self,
            input_tensor,
            output_tensor,
    ):
        shortcut = keras.layers.Conv1D(
            filters=int(output_tensor.shape[-1]),
            kernel_size=1,
            padding='same',
            use_bias=False,
        )(input_tensor)

        shortcut = keras.layers.BatchNormalization()(shortcut)

        output_tensor = keras.layers.Add()([shortcut, output_tensor])
        return keras.layers.Activation(activation='relu')(output_tensor)

    def evaluate(
            self,
            evaluate_generator: keras.utils.Sequence,
            verbose: int = 1,
    ):
        return self.model.evaluate(evaluate_generator, verbose=verbose)

    def predict(
            self,
            x_test: np.array,
            y_true: List[int],
            verbose: int = 1,
            return_metrics: bool = False,
    ):
        y_pred = [int(np.argmax(self.model.predict(
            x_test[np.newaxis, :, [1, i, i + 8, i + 16]],
            verbose=verbose,
        ))) for i in range(2, 10)]

        return utils.calculate_metrics(y_true, y_pred) if return_metrics else y_pred

    def save(self):
        path = os.path.join(utils.MODEL_DIR, self.model_name)
        self.model.save(filepath=path)
        return

    @staticmethod
    def load(config: Dict, model_name: str) -> 'InceptionTime':
        model = InceptionTime(**config)
        path = os.path.join(utils.MODEL_DIR, model_name)
        model.model = keras.models.load_model(path)
        return model


if __name__ == '__main__':
    for _dir in utils.DIRS:
        os.makedirs(_dir, exist_ok=True)

    _model = InceptionTime(**utils.TEST_MODEL_PARAMS)
    _model.compile()
    _model.summary()
