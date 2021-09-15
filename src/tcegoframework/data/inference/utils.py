import tcegoframework.config as config
from pandas import DataFrame


def initialize_inference_dict(data: DataFrame) -> DataFrame:
    config.INFERENCE_DICT = {}
    for i in range(0, data.shape[0]):
        empenho = data.iloc[i, :]
        config.INFERENCE_DICT[empenho['empenho_sequencial_empenho']] = {
            'Identificador': empenho['empenho_sequencial_empenho'],
            'Natureza Real': empenho['natureza_despesa_cod'],
            'Natureza Predita': None,
            'Corretude': None,
            'Resultado': None,
        }


def resolve_output(data, y1, y2):
    for i in range(data.shape[0]):
        key = data.iloc[i, :]['empenho_sequencial_empenho']
        config.INFERENCE_DICT[key]['Natureza Predita'] = y1[i]
        config.INFERENCE_DICT[key]['Corretude'] = y2[i]
        config.INFERENCE_DICT[key]['Resultado'] = resolve_result(
            config.INFERENCE_DICT[key]['Natureza Real'],
            config.INFERENCE_DICT[key]['Natureza Predita'],
            config.INFERENCE_DICT[key]['Corretude']
        )


def resolve_result(y_true, y_pred, y_pred_correctness) -> str:

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
