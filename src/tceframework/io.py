from os import makedirs, rename
from os.path import dirname
from typing import Any

import dill
from pandas import DataFrame, read_csv, read_excel
from torch import device, load, save
import json
import tceframework.config as config
import joblib


def load_csv_data(path: str) -> DataFrame:
    data = read_csv(path, encoding='utf-8', low_memory=False)
    return data


def save_csv_data(data: DataFrame, filename: str):
    filename = config.META_PATH + filename
    makedirs(dirname(filename), exist_ok=True)
    data.to_csv(filename)


def load_excel_data(path: str) -> DataFrame:
    data = read_excel(path)
    return data


def dump_encoder(encoder: Any, filename: str) -> None:
    filename = config.ENC_PATH + filename
    makedirs(dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        dill.dump(encoder, file)
        file.close()


def load_encoder(filename: str) -> Any:
    filename = config.ENC_PATH + filename
    with open(filename, 'rb') as file:
        encoder = dill.load(file)
        file.close()
    return encoder


def save_torch_model(modelstate: Any, filename: str) -> None:
    filename = config.MODEL_PATH + filename
    makedirs(dirname(filename), exist_ok=True)
    save(modelstate, filename)


def load_torch_model(filename: str, ) -> Any:
    filename = config.MODEL_PATH + filename
    state = load(filename, map_location=config.BERT_DEVICE)
    return state


def dump_model(model: Any, filename: str) -> None:
    filename = config.MODEL_PATH + filename
    makedirs(dirname(filename), exist_ok=True)
    joblib.dump(model, filename, compress=3)
    # with open(filename, 'wb') as file:
    #     dill.dump(model, file)
    #     file.close()


def load_model(filename: str) -> Any:
    filename = config.MODEL_PATH + filename
    model = joblib.load(filename)
    # with open(filename, 'rb') as file:
    #     model = dill.load(file)
    #     file.close()
    return model


def printfile(content: str, filename: str = 'out.txt') -> None:
    filename = config.STDOUT_REDIR_PATH + filename
    makedirs(dirname(filename), exist_ok=True)
    with open(filename, 'a') as file:
        print(content, file=file)
        file.close()


def change_root_dir_name(directory_name):
    rename(config.ROOT_DIR, directory_name)


def save_scope_dict(filename):
    filename = config.MODEL_PATH + filename
    makedirs(dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        dill.dump(config.CLASS_DICT, file)
        file.close()
    config.CLASS_DICT = None


def load_scope_dict(filename):
    filename = config.MODEL_PATH + filename
    with open(filename, 'rb') as file:
        config.CLASS_DICT = dill.load(file)
        file.close()


def save_inference_results(filename):
    result = DataFrame(config.INFERENCE_DICT)
    result.to_csv(filename)


def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)
        file.close()


def get_full_database():
    pass
