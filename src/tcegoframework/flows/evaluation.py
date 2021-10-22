import time
from functools import partial

from numpy import array
from pandas.core.frame import DataFrame
from sklearn.metrics import classification_report
from tcegoframework.cfgparsing import get_algorithm, get_validated_data_path
from tcegoframework.flows.inference import (inference_bert_rf_natureza,
                                            inference_rf_natureza,
                                            inference_svm_natureza)
from tcegoframework.io import load_excel_data
from tcegoframework.preprocessing.text import regularize_columns_name


def compute_agreement(y_true, y_pred) -> str:
    if y_true == y_pred:
        return 'OK'
    else:
        return 'INCONCLUSIVO'


def compute_output(data: DataFrame, inference_dict: dict, y_natureza: array, y_corretude: array) -> dict:
    for i in range(data.shape[0]):
        pass
    return inference_dict


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
