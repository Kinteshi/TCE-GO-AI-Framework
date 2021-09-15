from functools import partial
from typing import Iterable, Union

from pandas.core import series

import tcegoframework.config as config
from numpy.lib.function_base import iterable
from pandas import DataFrame
from tcegoframework.io import load_scope_dict


def scope_filter(data: DataFrame, scope_dict: dict) -> tuple[DataFrame, DataFrame]:
    mask = data.apply(partial(isinscope, scope_dict), axis=1)
    oos = data.loc[~mask, :].reset_index(drop=True)
    data = data.loc[mask, :].reset_index(drop=True)
    return data, oos


def isinscope(scope_dict: dict, empenho: series.Series) -> bool:
    if empenho['natureza_despesa_cod'] not in scope_dict:
        return False
    elif empenho['natureza_despesa_cod'] in scope_dict:
        key = empenho['natureza_despesa_cod']
        if scope_dict[key] != 'Em escopo':
            return False
        elif empenho['valor_saldo_do_empenho'] == 0:
            return False
        else:
            return True


def create_scope_dict(data: DataFrame) -> dict[str, str]:
    scope_dict = {}
    for target in data['natureza_despesa_cod'].unique():
        scope_dict[target] = 'Em escopo'
    return scope_dict


def change_scope_dict(scope_dict: dict[str, str], target_keys: Iterable[str], scope_status: str) -> dict[str, str]:
    for target_key in target_keys:
        scope_dict[target_key] = scope_status
    return scope_dict


def masked_filter(data: DataFrame, mask: list[bool]) -> tuple[DataFrame, DataFrame]:
    return data.loc[mask, :], data.loc[~mask, :]


def where_class_92(data: DataFrame) -> list[bool]:
    regex = r'\d[.]\d[.]\d\d[.]92[.]\d\d'
    mask = ~data['natureza_despesa_cod'].str.fullmatch(regex)
    return mask


def where_zero_value(data: DataFrame) -> Iterable[bool]:
    return data['valor_saldo_do_empenho'] > 0


def where_expired_class(data: DataFrame, expired_classes: DataFrame) -> Iterable[bool]:
    expired_classes = expired_classes['nat_despesa'].to_list()
    mask = ~data['natureza_despesa_cod'].isin(expired_classes)
    return mask


def where_below_threshold(data: DataFrame, threshold: int) -> Iterable[bool]:
    column = 'natureza_despesa_cod'
    counters = data[column].value_counts().to_frame()
    above_threshold = counters[counters[column] >= threshold].index.to_list()
    mask = data[column].isin(above_threshold)
    return mask
