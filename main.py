from inception import utils
from inception.datagen import DataGenerator
from inception.net import InceptionTime


if __name__ == '__main__':
    utils.create_dirs()

    _normalized_data = True
    _delete_old = True

    if _delete_old:
        import shutil
        shutil.rmtree(utils.MODEL_DIR)
        shutil.rmtree(utils.LOGS_DIR)
        utils.create_dirs()

        _model_num = 0
    else:
        _model_num = utils.increment_model_number(0)

    _data_paths = utils.NORMALIZED_DATA_PATHS if _normalized_data else utils.RAW_DATA_PATHS

    _train_gen = DataGenerator(_data_paths[0:3], **utils.TEST_DATAGEN_PARAMS)
    _valid_gen = DataGenerator(_data_paths[3:4], **utils.TEST_DATAGEN_PARAMS)

    _model = InceptionTime(f'test_model_{_model_num}', **utils.TEST_MODEL_PARAMS)
    _model.compile()
    _model.summary()
    _model.train(_train_gen, _valid_gen, **utils.TEST_TRAIN_PARAMS)
    _model.save()

    _test_gen = DataGenerator(_data_paths[4:5], **utils.TEST_DATAGEN_PARAMS)
    _eval = _model.evaluate(_test_gen)
    _line = ', '.join([f'{_p:.2f}' for _p in _eval])
    print(f'model {_model_num}, performance: {_line}')
