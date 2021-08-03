from typing import Iterable

import tceframework.config as config
from numpy.lib.function_base import iterable
from pandas import DataFrame
from tceframework.io import load_scope_dict


def initialize_class_dict(data: DataFrame) -> None:
    config.CLASS_DICT = {}
    for target in data['natureza_despesa_cod'].unique():
        config.CLASS_DICT[target] = 'Em escopo'


def change_scope(targets: Iterable, status: str) -> None:
    for target in targets:
        config.CLASS_DICT[target] = status


def scope_filter(data: DataFrame):
    config.CLASS_DICT = load_scope_dict('scope.pkl')

    # Filter out of scope dict -> Newly added classes - Unseen before
    mask = data.apply(blame, axis=1)
    data = data.loc[mask, :]
    data = data.reset_index(drop=True)

    return data

    # lambda empenho: return config.CLASS_DICT[empenho['natureza_despesa_cod']]


def blame(empenho):

    if empenho['natureza_despesa_cod'] not in config.CLASS_DICT:
        key = empenho['empenho_sequencial_empenho']
        config.INFERENCE_DICT[key]['Natureza Predita'] = 'Classe desconhecida'
        config.INFERENCE_DICT[key]['Corretude'] = 'Classe desconhecida'
        config.INFERENCE_DICT[key]['Resultado'] = 'Classe desconhecida'
        return False
    elif empenho['natureza_despesa_cod'] in config.CLASS_DICT:
        key = empenho['natureza_despesa_cod']
        if config.CLASS_DICT[key] != 'Em escopo':
            info = config.CLASS_DICT[key]
            key = empenho['empenho_sequencial_empenho']
            config.INFERENCE_DICT[key]['Natureza Predita'] = info
            config.INFERENCE_DICT[key]['Corretude'] = info
            config.INFERENCE_DICT[key]['Resultado'] = info
            return False
        else:
            return True
    elif empenho['valor_saldo_do_empenho'] == 0:
        key = empenho['empenho_sequencial_empenho']
        info = 'Saldo zerado'
        config.INFERENCE_DICT[key]['Natureza Predita'] = info
        config.INFERENCE_DICT[key]['Corretude'] = info
        config.INFERENCE_DICT[key]['Resultado'] = info
        return False


def min_docs_class(data: DataFrame, column: str, threshold: int) -> DataFrame:
    counters = data[column].value_counts().to_frame()
    above_threshold = counters[counters[column] >= threshold].index.to_list()
    mask = data[column].isin(above_threshold)
    change_scope(
        data.loc[~mask, 'natureza_despesa_cod'],
        f'Classe abaixo do requerimento de {threshold} documentos')
    data = data.loc[mask, :]
    data = data.reset_index(drop=True)
    return data


def remove_class_92(data: DataFrame) -> DataFrame:
    regex = r'\d[.]\d[.]\d\d[.]92[.]\d\d'
    mask = ~data['natureza_despesa_cod'].str.fullmatch(regex)
    change_scope(
        data.loc[~mask, 'natureza_despesa_cod'],
        f'Classe 92')
    data = data.loc[mask, :]
    data = data.reset_index(drop=True)
    return data


def remove_expired_classes(data: DataFrame, expired_data: DataFrame) -> DataFrame:
    expired_classes = expired_data['nat_despesa'].to_list()
    mask = ~data['natureza_despesa_cod'].isin(expired_classes)
    change_scope(
        data.loc[~mask, 'natureza_despesa_cod'],
        f'Classe fora de vigência')
    data = data.loc[mask, :]
    data = data.reset_index(drop=True)
    return data


def remove_zeroed_documents(data: DataFrame) -> DataFrame:
    mask = data['valor_saldo_do_empenho'] > 0
    data = data.loc[mask, :]
    data = data.reset_index(drop=True)
    return data
