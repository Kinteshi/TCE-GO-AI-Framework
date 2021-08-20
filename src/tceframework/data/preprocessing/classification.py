import copy
from datetime import datetime
from typing import Any, Union

from numpy import array, zeros
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from tceframework.data.misc import create_data_loader
from tceframework.data.preprocessing.encoders import generate_fit_ohe, generate_fit_scaler, generate_fit_tfidf, generate_ohe, generate_scaler, generate_tfidf
from tceframework.data.text import clean_nlp, clean_tfidf
from tceframework.io import dump_encoder, load_encoder
from torch.utils.data import dataloader
from transformers import BertTokenizer
import tceframework.config as config
from tceframework.model.bert import get_representation, get_saved_model
from dateutil.parser import parse


def pp_bert_training(data: DataFrame) -> Union[Any, Any]:

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


def get_n_classes(data: DataFrame, column: str) -> int:
    n_classes = data[column].value_counts().shape[0]
    return n_classes


def pp_bert_inference(data: DataFrame) -> DataFrame:

    data = data[['empenho_historico', 'natureza_despesa_cod']]
    data['empenho_historico'].update(data['empenho_historico'].map(clean_nlp))
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

    encoder = load_encoder('targetNLP.pkl')
    data['natureza_despesa_cod'].update(
        encoder.fit_transform(data['natureza_despesa_cod']))

    dataloader = create_data_loader(data, tokenizer)

    return dataloader


def pp_tabular_training(data: DataFrame):

    target = data.natureza_despesa_cod

    data = data.loc[:, (*config.CLF_CAT,
                        *config.CLF_TEXT,
                        *config.CLF_NUM,)]

    data, categorical_columns, numerical_columns = data_preparation(
        data,
        categorical_columns=config.CLF_CAT,
        numerical_columns=config.CLF_NUM,)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3, random_state=config.RANDOM_SEED, stratify=target)

    X_train, X_test = encode_train_test(X_train,
                                        X_test,
                                        numerical_columns,
                                        categorical_columns,
                                        config.CLF_TEXT,
                                        tfidf=False,
                                        bert=True,
                                        y_train=y_train,
                                        y_test=y_test,
                                        prefix='clf')

    return X_train, X_test, y_train, y_test


def pp_second_tabular_training(data: DataFrame):

    target = data.analise

    data = data.loc[:, (*config.CLF2_CAT,
                        *config.CLF2_TEXT,
                        *config.CLF2_NUM,)]

    data, categorical_columns, numerical_columns = data_preparation(
        data,
        categorical_columns=config.CLF2_CAT,
        numerical_columns=config.CLF2_NUM,)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=config.RANDOM_SEED, stratify=target)

    X_train, X_test = encode_train_test(X_train,
                                        X_test,
                                        numerical_columns,
                                        categorical_columns,
                                        config.CLF2_TEXT,
                                        tfidf=True,
                                        bert=False,
                                        y_train=y_train,
                                        y_test=y_test,
                                        prefix='cr')

    return X_train, X_test, y_train, y_test


def code_reaper(input_text):
    return input_text[7:]


def date_to_month(input_date):
    if isinstance(input_date, int):
        date = datetime.fromordinal(
            datetime(1900, 1, 1).toordinal() +
            input_date -
            2)
    else:
        date = parse(input_date)
    return date.month


def data_preparation(data: DataFrame, categorical_columns, numerical_columns):

    data = data.reset_index(drop=True)
    # Criação de meta-atributos e tratamento comum para todos os dados

    # Criação do meta-atributo mês
    if 'periodo' in categorical_columns:
        data['periodo'].update(data['periodo'].map(date_to_month))

    # Criação do meta-atributo "Pessoa Jurídica?"
    pessoa_juridica = array(
        [1 if cpf == '-' else 0
            for cpf in data['beneficiario_cpf'].values])
    data['pessoa_juridica'] = pessoa_juridica.astype('int8')

    data.drop(['beneficiario_cpf', 'beneficiario_cnpj',
               'beneficiario_cpf/cnpj'], axis='columns', inplace=True)
    categorical_columns.remove('beneficiario_cpf')
    categorical_columns.remove('beneficiario_cnpj')
    categorical_columns.remove('beneficiario_cpf/cnpj')
    categorical_columns.append('pessoa_juridica')
    del pessoa_juridica

    # Codigo que gera o meta atributo "orgao_sucedido" onde 1 representa que o orgao tem um novo orgao sucessor e 0 caso contrario
    orgao_sucedido = zeros(data.shape[0])

    for i in range(data.shape[0]):
        if(data['orgao'].iloc[i] != data['orgao_sucessor_atual'].iloc[i]):
            orgao_sucedido[i] = 1

    data['orgao_sucedido'] = orgao_sucedido.astype('int8')
    data.drop(['orgao'], axis='columns', inplace=True)
    categorical_columns.remove('orgao')
    categorical_columns.append('orgao_sucedido')
    del orgao_sucedido

    # Codigo que retira o codigo de programa (retirando 10 valores)
    data['programa'] = data['programa'].map(code_reaper)
    # Codigo que retira o codigo de acao (retirando 77 valores)
    data['acao'] = data['acao'].map(code_reaper)

    # Codigo que concatena acao e programa
    acao_programa = zeros(data.shape[0], dtype='object')
    for i in range(data.shape[0]):
        acao_programa[i] = (data['acao'].iloc[i] + ' & ' +
                            data['programa'].iloc[i])
    data['acao_programa'] = acao_programa
    data.drop(['acao', 'programa'], axis='columns', inplace=True)
    categorical_columns.remove('acao')
    categorical_columns.remove('programa')
    categorical_columns.append('acao_programa')
    del acao_programa

    return data, categorical_columns, numerical_columns


def generate_fit_process_count(data):
    process_count = data['empenho_numero_do_processo'].value_counts()
    process_count = process_count.to_dict()

    empenhos_processo = zeros(data.shape[0])

    for i in range(data.shape[0]):
        empenhos_processo[i] = process_count[data['empenho_numero_do_processo'].iloc[i]]
    data['empenhos_por_processo'] = empenhos_processo
    data.drop('empenho_numero_do_processo', axis='columns', inplace=True)
    dump_encoder(process_count, f'enc_empenhos_por_processo.pkl')

    return data


def generate_process_count(data):

    process_count = load_encoder(f'enc_empenhos_por_processo.pkl')

    empenhos_processo = zeros(data.shape[0])

    for i in range(data.shape[0]):
        key = data['empenho_numero_do_processo'].iloc[i]
        if key in process_count:
            empenhos_processo[i] = process_count[key]
        else:
            empenhos_processo[i] = 0
    data['empenhos_por_processo'] = empenhos_processo

    data.drop('empenho_numero_do_processo', axis='columns', inplace=True)

    return data


def generate_bert_representation(X, y, columns, model, encoder) -> DataFrame:
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    for col_name in columns:
        X[col_name].update(X[col_name].map(clean_nlp))
        data = DataFrame()
        data[col_name] = X[col_name].reset_index(drop=True)
        data['natureza_despesa_cod'] = y.reset_index(drop=True)
        data['natureza_despesa_cod'].update(
            encoder.transform(data['natureza_despesa_cod']))
        data_loader = create_data_loader(
            data, tokenizer)
        representation = get_representation(model, data_loader)
        X.drop([col_name], inplace=True, axis='columns')
        X = X.join(representation)
    return X


def encode_train_test(X_train: DataFrame, X_test: DataFrame, numerical_columns: list, categorical_columns: list, text_columns: list, tfidf: bool, bert: bool, y_train, y_test, prefix=''):
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    if 'empenho_numero_do_processo' in categorical_columns:
        X_train = generate_fit_process_count(X_train)
        X_test = generate_process_count(X_test)
        categorical_columns.remove('empenho_numero_do_processo')
        numerical_columns.append('empenhos_por_processo')

    X_train = generate_fit_scaler(X_train, numerical_columns, prefix)
    X_test = generate_scaler(X_test, numerical_columns, prefix)

    X_train = generate_fit_ohe(X_train, categorical_columns, prefix)
    X_test = generate_ohe(X_test, categorical_columns, prefix)

    if tfidf:
        X_train = generate_fit_tfidf(X_train, text_columns, prefix)
        X_test = generate_tfidf(X_test, text_columns, prefix)
    elif bert:
        label_encoder = load_encoder('targetNLP.pkl')
        model = get_saved_model(len(label_encoder.classes_))
        X_train = generate_bert_representation(
            X_train, y_train, text_columns, model, label_encoder)
        X_test = generate_bert_representation(
            X_test, y_test, text_columns, model, label_encoder)

    return X_train, X_test


def encode_inference(X, y, numerical_columns: list, categorical_columns: list, text_columns: list, tfidf: bool, bert: bool, prefix=''):
    X = X.reset_index(drop=True)

    if 'empenho_numero_do_processo' in categorical_columns:
        X = generate_process_count(X)
        categorical_columns.remove('empenho_numero_do_processo')
        numerical_columns.append('empenhos_por_processo')

    X = generate_scaler(X, numerical_columns, prefix)

    X = generate_ohe(X, categorical_columns, prefix)

    if tfidf:
        X = generate_tfidf(X, text_columns, prefix)
    elif bert:
        label_encoder = load_encoder('targetNLP.pkl')
        model = get_saved_model(len(label_encoder.classes_))
        X = generate_bert_representation(
            X, y, text_columns, model, label_encoder)

    return X


def pp_tabular_inference(data):

    target = data.natureza_despesa_cod
    data = data.loc[:, (*config.CLF_CAT,
                        *config.CLF_TEXT,
                        *config.CLF_NUM,)]

    data, categorical_columns, numerical_columns = data_preparation(
        data, categorical_columns=config.CLF_CAT, numerical_columns=config.CLF_NUM,)

    X = encode_inference(data, target, numerical_columns, categorical_columns,
                         config.CLF_TEXT, tfidf=False, bert=True, prefix='clf')

    return X


def pp_second_tabular_inference(data: DataFrame):

    data = data.loc[:, (*config.CLF2_CAT,
                        *config.CLF2_TEXT,
                        *config.CLF2_NUM,)]

    data, categorical_columns, numerical_columns = data_preparation(
        data,
        categorical_columns=config.CLF2_CAT,
        numerical_columns=config.CLF2_NUM,)

    X = encode_inference(
        data,
        None,
        numerical_columns,
        categorical_columns,
        config.CLF2_TEXT,
        tfidf=True,
        bert=False,
        prefix='cr')

    return X
