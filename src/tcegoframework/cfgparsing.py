from configparser import ConfigParser
from typing import Union

PARSER = ConfigParser()
PARSER.read('config.ini')

GENERAL = 'options.general'
DREMIO = 'options.dremio'
TRAINING = 'options.training'
INFERENCE = 'options.inference'
HDFS = 'options.hdfs'


# [options.general]

def get_random_seed() -> int:
    return PARSER.getint(GENERAL, 'random_seed', fallback=15)


# [options.dremio]


def get_dremio_connection() -> str:
    return PARSER.get(DREMIO, 'connection')


def get_dremio_user() -> str:
    return PARSER.get(DREMIO, 'user')


def get_dremio_password() -> str:
    return PARSER.get(DREMIO, 'password')


# [options.training]


def get_training_dataset_path() -> Union[str, None]:
    return PARSER.get(TRAINING, 'dataset_path', fallback=None)


def get_training_sampling_number() -> Union[int, None]:
    return PARSER.getint(TRAINING, 'sample_dataset', fallback=None)


def get_expired_labels_path() -> str:
    return PARSER.get(TRAINING, 'expired_class_path')


def get_validated_data_path() -> str:
    return PARSER.get(TRAINING, 'validated_data_path')


def get_label_population_floor() -> int:
    return PARSER.getint(TRAINING, 'min_documents_class', fallback=200)


def get_algorithm() -> str:
    return PARSER.get(TRAINING, 'algorithm', fallback='bert_rf')


def get_epochs() -> int:
    return PARSER.getint(TRAINING, 'epochs', fallback=5)

# [options.inference]


def get_inference_dataset_path() -> Union[str, None]:
    return PARSER.get(INFERENCE, 'dataset_path', fallback=None)


# [options.hdfs]

def use_hdfs() -> bool:
    return PARSER.getboolean(HDFS, 'use', fallback=True)


def get_hdfs_domain() -> str:
    return PARSER.get(HDFS, 'domain')


def get_hdfs_url() -> str:
    return PARSER.get(HDFS, 'url')


def get_hdfs_port() -> int:
    return PARSER.getint(HDFS, 'port')


def get_hdfs_user() -> str:
    return PARSER.get(HDFS, 'user')


def get_hdfs_password() -> str:
    return PARSER.get(HDFS, 'password')


def get_hdfs_dir_path() -> str:
    return PARSER.get(HDFS, 'path')
