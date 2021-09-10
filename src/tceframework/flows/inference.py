import warnings
from numpy import array

from pandas.core.frame import DataFrame

import tceframework.config as config
from tceframework.data.filter import scope_filter
from tceframework.dremio import construct_query, execute_query
from tceframework.io import load_model, load_scope_dict, save_inference_results, save_json
from datetime import datetime
from tceframework.preprocessing.classification import preprocessing_inference_corretude, preprocessing_inference_natureza

from tceframework.preprocessing.text import regularize_columns_name

warnings.filterwarnings('ignore')


def query_dataset(filters: dict) -> DataFrame:
    query = construct_query(filters)
    data = execute_query(query)
    return data


def create_inference_dict(data: DataFrame) -> dict:
    inference_dict = {}
    for i in range(0, data.shape[0]):
        empenho = data.iloc[i, :]
        inference_dict[empenho['empenho_sequencial_empenho']] = {
            'Identificador': empenho['empenho_sequencial_empenho'],
            'Natureza Real': empenho['natureza_despesa_cod'],
            'Natureza Predita': None,
            'Corretude': None,
            'Resultado': None,
        }
    return inference_dict


def change_inference_dict(out_of_scope: DataFrame, scope_dict: dict, inference_dict: dict) -> dict:
    for _, empenho in out_of_scope.iterrows():
        if empenho['natureza_despesa_cod'] not in scope_dict:
            key = empenho['empenho_sequencial_empenho']
            info = 'Classe desconhecida'
            inference_dict[key]['Natureza Predita'] = info
            inference_dict[key]['Corretude'] = info
            inference_dict[key]['Resultado'] = 'INC'
        elif empenho['natureza_despesa_cod'] in scope_dict:
            key = empenho['natureza_despesa_cod']
            if scope_dict[key] != 'Em escopo':
                info = scope_dict[key]
                key = empenho['empenho_sequencial_empenho']
                inference_dict[key]['Natureza Predita'] = info
                inference_dict[key]['Corretude'] = info
                inference_dict[key]['Resultado'] = 'INC'
            elif empenho['valor_saldo_do_empenho'] == 0:
                key = empenho['empenho_sequencial_empenho']
                info = 'Saldo zerado'
                inference_dict[key]['Natureza Predita'] = info
                inference_dict[key]['Corretude'] = info
                inference_dict[key]['Resultado'] = 'SE0'
    return inference_dict


def inference_svm_natureza(data: DataFrame) -> list:
    X = preprocessing_inference_natureza(data.copy())
    model = load_model(filename='svm_natureza_above_model')
    y_proba_above = model.predict_proba(X)
    y_pred_above = model.predict(X)
    model = load_model(filename='svm_natureza_below_model')
    y_proba_below = model.predict_proba(X)
    y_pred_below = model.predict(X)
    y_pred = [
        a if probA >= probB else b
        for a, b, probA, probB in
        zip(y_pred_above, y_pred_below, y_proba_above, y_proba_below)
    ]
    return array(y_pred)


def inference_rf_natureza(data: DataFrame) -> list:
    X = preprocessing_inference_natureza(data.copy())
    model = load_model(filename='rf_natureza_above_model')
    y_proba_above = model.predict_proba(X)
    y_pred_above = model.predict(X)
    model = load_model(filename='rf_natureza_below_model')
    y_proba_below = model.predict_proba(X)
    y_pred_below = model.predict(X)
    y_pred = [
        a if probA >= probB else b
        for a, b, probA, probB in
        zip(y_pred_above, y_pred_below, y_proba_above, y_proba_below)
    ]
    return array(y_pred)


def inference_corretude(data: DataFrame) -> array:
    X = preprocessing_inference_corretude(data.copy())
    model = load_model(filename='random_forest_ii_model.pkl')
    y_pred_corretude = model.predict(X)
    return y_pred_corretude


def compute_result_code(y_true, y_pred, y_pred_correctness) -> str:
    if y_true == y_pred:
        if y_pred_correctness == 'INCORRETO':
            return 'INCV_M2'
        elif y_pred_correctness == 'OK':
            return 'C_M1-M2'
        elif y_pred_correctness == 'INCONCLUSIVO':
            return 'AD_M2'
    else:
        if y_pred_correctness == 'INCORRETO':
            return 'INCT_M1-M2'
        elif y_pred_correctness == 'OK':
            return 'INCV_M1'
        elif y_pred_correctness == 'INCONCLUSIVO':
            return 'INCV_M1-AD_M2'


def compute_output(data: DataFrame, inference_dict: dict, y_natureza: array, y_corretude: array) -> dict:
    for i in range(data.shape[0]):
        key = data.iloc[i, :]['empenho_sequencial_empenho']
        inference_dict[key]['Natureza Predita'] = y_natureza[i]
        inference_dict[key]['Corretude'] = y_corretude[i]
        inference_dict[key]['Resultado'] = compute_result_code(
            inference_dict[key]['Natureza Real'],
            inference_dict[key]['Natureza Predita'],
            inference_dict[key]['Corretude']
        )
    return inference_dict


def get_algorithm() -> str:
    return config.PARSER.get(
        'options.training',
        'algorithm',
        fallback='svm'
    )


def inference_flow(filters: dict):
    # Executing query
    data = query_dataset(filters)
    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)

    inference_dict = create_inference_dict(data)
    scope_dict = load_scope_dict('scope.pkl')
    data, out_of_scope = scope_filter(data, scope_dict)

    inference_dict = change_inference_dict(
        out_of_scope, scope_dict, inference_dict)

    if algorithm := get_algorithm() == 'svm':
        inference_natureza = inference_svm_natureza
    elif algorithm == 'rf':
        inference_natureza = inference_rf_natureza

    y_pred_natureza = inference_natureza(data.copy())
    y_pred_corretude = inference_corretude(data.copy())

    inference_dict = compute_output(
        data, inference_dict, y_pred_natureza, y_pred_corretude)

    date = datetime.today().strftime('%d-%m-%Y')

    save_inference_results(f'{date}_results.csv', inference_dict)