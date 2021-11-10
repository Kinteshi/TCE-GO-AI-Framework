import time
from functools import partial

from numpy import array
from pandas.core.frame import DataFrame
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tcegoframework import config
from tcegoframework.cfgparsing import get_algorithm, get_validated_data_path
from tcegoframework.flows.inference import (inference_bert_rf_natureza,
                                            inference_corretude,
                                            inference_rf_natureza,
                                            inference_svm_natureza)
from tcegoframework.io import load_excel_data
from tcegoframework.model.metrics import classification_report_csv
from tcegoframework.preprocessing.text import regularize_columns_name


def evaluation_flow():
    # Executing query
    print('Preparando e executando avaliação...')
    data = load_excel_data(get_validated_data_path())
    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)

    if get_algorithm() == 'svm':
        inference_natureza = partial(inference_svm_natureza)
    elif get_algorithm() == 'rf':
        inference_natureza = partial(inference_rf_natureza)
    elif get_algorithm() == 'bert_rf':
        inference_natureza = partial(inference_bert_rf_natureza)

    for label in ['OK', 'INCONCLUSIVO', 'INCORRETO']:
        temp_data = data.loc[data.analise == label]

        print(f'Inferência de Natureza para documentos {label}...')
        time_ref = time.time()
        y_pred_natureza = inference_natureza(temp_data.copy())

        y_true = temp_data.natureza_despesa_cod

        report = classification_report(
            y_true, y_pred_natureza, output_dict=True)
        report = DataFrame(report).transpose()
        report.to_csv(f'{label}_eval_report.csv')
        print(f'Duração total: {(time.time() - time_ref)/60}')

    print('Finalizado.')


def training_evaluation_flow():
    # Executing query
    print('Preparando e executando avaliação...')
    data = load_excel_data(get_validated_data_path())
    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)

    _, data = train_test_split(
        data,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=data.analise)

    if get_algorithm() == 'svm':
        inference_natureza = partial(inference_svm_natureza)
    elif get_algorithm() == 'rf':
        inference_natureza = partial(inference_rf_natureza)
    elif get_algorithm() == 'bert_rf':
        inference_natureza = partial(inference_bert_rf_natureza)

    print('Avaliação do classificador de naturezas...')

    for label in ['OK', 'INCONCLUSIVO', 'INCORRETO']:
        temp_data = data.loc[data.analise == label]

        print(f'Inferência de Natureza para documentos {label}...')
        time_ref = time.time()
        y_pred_natureza = inference_natureza(temp_data.copy())

        y_true = temp_data.natureza_despesa_cod

        classification_report_csv(
            y_true, y_pred_natureza, False, f'natureza_{label}_eval_report.csv')
        print(f'Duração total: {(time.time() - time_ref)/60}')

    print('Avaliação do classificador de corretude...')
    time_ref = time.time()
    y_pred = inference_corretude(data.copy())
    y_true = data.analise
    classification_report_csv(
        y_true, y_pred, False, 'corretude_eval_report.csv')
    print(f'Duração total: {(time.time() - time_ref)/60}')

    print('Finalizado.')
