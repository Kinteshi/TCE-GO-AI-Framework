import json
from os import makedirs, rename
from os.path import dirname
from typing import Any, Union

import dill
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, read_csv, read_excel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import device, load, save

import tcegoframework.config as config
from tcegoframework.cfgparsing import (get_hdfs_dir_path, get_hdfs_domain,
                                       get_hdfs_password, get_hdfs_port,
                                       get_hdfs_url, get_hdfs_user, use_hdfs)
from tcegoframework.tcehdfs import TCEHDFS


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


def load_model(filename: str) -> Union[SVC, RandomForestClassifier]:
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


def save_scope_dict(filename: str, scope_dict: dict[str, str]) -> None:
    filename = config.MODEL_PATH + filename
    makedirs(dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        dill.dump(scope_dict, file)
        file.close()


def load_scope_dict(filename):
    filename = config.MODEL_PATH + filename
    with open(filename, 'rb') as file:
        class_dict = dill.load(file)
        file.close()
    return class_dict


def save_inference_results(filename: str, inference_dict: dict) -> DataFrame:
    result = DataFrame(inference_dict).transpose()
    result.to_csv(filename)

    if use_hdfs():
        try:
            hdfs = TCEHDFS(
                get_hdfs_domain(),
                get_hdfs_url(),
                get_hdfs_port(),
                get_hdfs_user(),
                get_hdfs_password()
            )

            hdfs.writeHDFS(
                get_hdfs_dir_path(),
                filename,
                result
            )
        except Exception as e:
            print('Ocorreu um erro ao salvar os resultados no HDFS. Erro: ', e)

    return result


def save_inference_plot(filename: str, data: DataFrame) -> None:
    sns.set()
    sns.set_context('talk')
    plt.figure(figsize=(15, 10))
    sns.countplot(y='Resultado', data=data)
    plt.title('Resumo dos resultados')
    plt.savefig(filename)


def save_bert_history_plot(history: dict, section: str) -> None:

    sns.set()
    sns.set_context('talk')
    plt.figure(figsize=(15, 10))

    plt.cla()
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Training history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f'{config.META_PATH}/{section}_loss.png')

    plt.cla()
    plt.plot(history['train_macro'], label='train macro')
    plt.plot(history['val_macro'], label='validation macro')
    plt.title('Training history')
    plt.ylabel('Macro')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f'{config.META_PATH}/{section}_macro.png')

    plt.cla()
    plt.plot(history['train_micro'], label='train micro')
    plt.plot(history['val_micro'], label='validation micro')
    plt.title('Training history')
    plt.ylabel('Micro')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f'{config.META_PATH}/{section}_micro.png')


def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)
        file.close()
