from copy import deepcopy

from pandas import DataFrame
from pandas.core.series import Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tcegoframework import config
from tcegoframework.data.misc import create_data_loader
from tcegoframework.io import dump_encoder, load_encoder
from tcegoframework.preprocessing.encoders import (encode_inference,
                                                   encode_train_test)
from tcegoframework.preprocessing.text import clean_nlp
from tcegoframework.preprocessing.transform import data_preparation
from torch.utils.data import dataloader
from transformers import BertTokenizer


def preprocessing_training_natureza(data: DataFrame, text_representation: str, section: str) -> tuple[
        DataFrame, DataFrame, Series, Series]:
    target = data.natureza_despesa_cod

    cat_col = config.CLF_CAT.copy()
    text_col = config.CLF_TEXT.copy()
    num_col = config.CLF_NUM.copy()

    data = data.loc[:, (*cat_col,
                        *text_col,
                        *num_col,)]

    data, categorical_columns, numerical_columns = data_preparation(
        data,
        categorical_columns=cat_col,
        numerical_columns=num_col, )

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=config.RANDOM_SEED, stratify=target)

    X_train, X_test = encode_train_test(X_train,
                                        X_test,
                                        numerical_columns,
                                        categorical_columns,
                                        text_col,
                                        y_train=y_train,
                                        y_test=y_test,
                                        prefix=f'nat_{section}',
                                        text_representation=text_representation)

    return X_train, X_test, y_train, y_test


def preprocessing_inference_natureza(data: DataFrame, text_representation: str, section: str) -> DataFrame:
    target = data.natureza_despesa_cod

    cat_col = [
        # 'exercicio_do_orcamento_ano',
        # 'empenho_sequencial_empenho',
        'orgao',
        'orgao_sucessor_atual',
        'tipo_administracao_nome',
        'tipo_poder_nome',
        # 'classificacao_orcamentaria_descricao',
        'funcao',
        'subfuncao',
        'programa',
        'acao',
        # 'grupo_despesa',
        # 'elemento_despesa',
        # 'natureza_despesa_cod',
        # 'natureza_despesa_nome',
        # 'formalidade_nome',
        'modalidade_licitacao_nome',
        # 'fonte_recurso_cod',
        'fonte_recurso_nome',
        'beneficiario_cnpj',
        'beneficiario_cpf',
        'beneficiario_cpf/cnpj',
        'periodo',
        'empenho_numero_do_processo',
        # 'empenho_sequencial_empenho.1',
    ]
    text_col = [
        # 'beneficiario_nome',
        'empenho_historico',
    ]
    num_col = [
        'valor_empenhado',
        'valor_anulacao_empenho',
        # 'valor_estorno_anulacao_empenho',
        'valor_cancelamento_empenho',
        # 'valor_anulacao_cancelamento_empenho',
        'valor_saldo_do_empenho',
        'valor_liquidacao_empenho',
        'valor_anulacao_liquidacao_empenho',
        'valor_saldo_liquidado',
        'valor_ordem_de_pagamento',
        'valor_guia_recolhimento',
        'valor_anulacao_ordem_de_pagamento',
        'valor_estorno_anulacao_o_pagamento',
        'valor_estorno_guia_recolhimento',
        'valor_saldo_pago',
        'valor_saldo_a_pagar',
        'valor_a_liquidar',
        'valor_a_pagar_liquidado'
    ]

    data = data.loc[:, (*cat_col,
                        *text_col,
                        *num_col,)]

    data, categorical_columns, numerical_columns = data_preparation(
        data, categorical_columns=cat_col, numerical_columns=config.CLF_NUM, )

    X = encode_inference(data, target, numerical_columns, categorical_columns,
                         config.CLF_TEXT, prefix=f'nat_{section}', text_representation=text_representation)

    return X


def preprocessing_training_natureza_bert(data: DataFrame) -> tuple[dataloader.DataLoader, dataloader.DataLoader]:
    data = data[['empenho_historico', 'natureza_despesa_cod']]
    data['empenho_historico'].update(data['empenho_historico'].map(clean_nlp))
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

    df_train, df_test = train_test_split(
        data,
        test_size=0.3,
        random_state=config.RANDOM_SEED,
        stratify=data['natureza_despesa_cod']
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    encoder = LabelEncoder()
    df_train['natureza_despesa_cod'].update(
        encoder.fit_transform(df_train['natureza_despesa_cod']))
    df_test['natureza_despesa_cod'].update(
        encoder.transform(df_test['natureza_despesa_cod']))
    dump_encoder(encoder, 'targetNLP.pkl')

    traindl = create_data_loader(df_train, tokenizer)
    testdl = create_data_loader(df_test, tokenizer)

    return traindl, testdl


def preprocessing_inference_natureza_bert(data: DataFrame) -> dataloader.DataLoader:
    data = data[['empenho_historico', 'natureza_despesa_cod']]
    data['empenho_historico'].update(data['empenho_historico'].map(clean_nlp))
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

    encoder = load_encoder('targetNLP.pkl')
    data['natureza_despesa_cod'].update(
        encoder.fit_transform(data['natureza_despesa_cod']))

    dataloader = create_data_loader(data, tokenizer)

    return dataloader


def preprocessing_training_corretude(data: DataFrame) -> tuple[DataFrame, DataFrame, Series, Series]:
    target = data.analise

    data = data.loc[:, (*config.CLF2_CAT,
                        *config.CLF2_TEXT,
                        *config.CLF2_NUM,)]

    data, categorical_columns, numerical_columns = data_preparation(
        data,
        categorical_columns=config.CLF2_CAT,
        numerical_columns=config.CLF2_NUM, )

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=config.RANDOM_SEED, stratify=target)

    X_train, X_test = encode_train_test(X_train,
                                        X_test,
                                        numerical_columns,
                                        categorical_columns,
                                        config.CLF2_TEXT,
                                        y_train=y_train,
                                        y_test=y_test,
                                        prefix='cr',
                                        text_representation='tfidf')

    return X_train, X_test, y_train, y_test


def preprocessing_inference_corretude(data: DataFrame) -> DataFrame:
    data = data.loc[:, (*config.CLF2_CAT,
                        *config.CLF2_TEXT,
                        *config.CLF2_NUM,)]

    data, categorical_columns, numerical_columns = data_preparation(
        data,
        categorical_columns=config.CLF2_CAT,
        numerical_columns=config.CLF2_NUM, )

    X = encode_inference(
        data,
        None,
        numerical_columns,
        categorical_columns,
        config.CLF2_TEXT,
        prefix='cr',
        text_representation='tfidf')

    return X
