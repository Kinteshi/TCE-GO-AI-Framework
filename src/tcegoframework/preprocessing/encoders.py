from typing import List, Union
from numpy import zeros, array
from pandas import DataFrame
from pandas.core.series import Series
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tcegoframework import config
from tcegoframework.data.misc import create_data_loader
from tcegoframework.io import dump_encoder, load_encoder
from tcegoframework.model.bert import (generate_bert_representation,
                                       get_saved_model)
from tcegoframework.preprocessing.text import clean_nlp, clean_tfidf
from transformers import BertTokenizer


def generate_fit_scaler(X_train, columns, prefix='') -> List[array]:
    out = []
    for col_name in columns:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X_train[col_name].values.reshape(-1, 1))
        out.append(scaled)
        dump_encoder(scaler, f'{prefix}_sc_{col_name}.pkl')
        del scaler
    return out


def generate_scaler(X_test, columns, prefix='') -> List[array]:
    out = []
    for col_name in columns:
        scaler = load_encoder(f'{prefix}_sc_{col_name}.pkl')
        scaled = scaler.transform(X_test[col_name].values.reshape(-1, 1))
        out.append(scaled)
    return out


def generate_fit_ohe(X_train, columns, prefix='') -> List[csr_matrix]:
    out = []
    for col_name in columns:
        ohe = OneHotEncoder(handle_unknown='ignore')
        enc = ohe.fit_transform(X_train[col_name].values.reshape(-1, 1))
        out.append(enc)
        dump_encoder(ohe, f'{prefix}_ohe_{col_name}.pkl')
        del ohe
    return out


def generate_ohe(X_test, columns, prefix='') -> List[csr_matrix]:
    out = []
    for col_name in columns:
        ohe = load_encoder(f'{prefix}_ohe_{col_name}.pkl')
        enc = ohe.transform(X_test[col_name].values.reshape(-1, 1))
        out.append(enc)
    return out


def generate_fit_tfidf(X_train, columns, prefix='') -> List[csr_matrix]:
    out = []
    for col_name in columns:
        X_train[col_name].update(X_train[col_name].map(clean_tfidf))
        tfv = TfidfVectorizer()
        tfidf = tfv.fit_transform(X_train[col_name])
        out.append(tfidf)
        dump_encoder(tfv, f'{prefix}_tfv_{col_name}.pkl')
        del tfv
    return out


def generate_tfidf(X_test, columns, prefix='') -> List[csr_matrix]:
    out = []
    for col_name in columns:
        X_test[col_name].update(X_test[col_name].map(clean_tfidf))
        tfv = load_encoder(f'{prefix}_tfv_{col_name}.pkl')
        tfidf = tfv.transform(X_test[col_name])
        out.append(tfidf)
    return out


def generate_fit_process_count(data) -> DataFrame:
    process_count = data['empenho_numero_do_processo'].value_counts()
    process_count = process_count.to_dict()

    empenhos_processo = zeros(data.shape[0])

    for i in range(data.shape[0]):
        empenhos_processo[i] = process_count[data['empenho_numero_do_processo'].iloc[i]]
    data['empenhos_por_processo'] = empenhos_processo
    data.drop('empenho_numero_do_processo', axis='columns', inplace=True)
    dump_encoder(process_count, f'enc_empenhos_por_processo.pkl')

    return data


def generate_process_count(data) -> DataFrame:
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


def generate_bert(X, y, columns, model, encoder) -> List[array]:
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    out = []
    for col_name in columns:
        X[col_name].update(X[col_name].map(clean_nlp))
        data = DataFrame()
        data[col_name] = X[col_name].reset_index(drop=True)
        data['natureza_despesa_cod'] = y.reset_index(drop=True)
        data['natureza_despesa_cod'].update(
            zeros(shape=data['natureza_despesa_cod'].shape))
        data_loader = create_data_loader(
            data, tokenizer)
        representation = generate_bert_representation(model, data_loader)
        out.append(representation.values)
    return out


def encode_train_test(
        X_train: DataFrame,
        X_test: DataFrame,
        numerical_columns: list,
        categorical_columns: list,
        text_columns: list,
        y_train: Series,
        y_test: Series,
        prefix: str,
        text_representation: str,
        section: str = None) -> Union[csr_matrix, csr_matrix]:
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    if 'empenho_numero_do_processo' in categorical_columns:
        X_train = generate_fit_process_count(X_train)
        X_test = generate_process_count(X_test)
        categorical_columns.remove('empenho_numero_do_processo')
        numerical_columns.append('empenhos_por_processo')

    scaled_numeric_train = generate_fit_scaler(
        X_train, numerical_columns, prefix)
    scaled_numeric_test = generate_scaler(X_test, numerical_columns, prefix)

    categorical_dummies_train = generate_fit_ohe(
        X_train, categorical_columns, prefix)
    categorical_dummies_test = generate_ohe(
        X_test, categorical_columns, prefix)

    if text_representation == 'tfidf':
        text_representation_train = generate_fit_tfidf(
            X_train, text_columns, prefix)
        text_representation_test = generate_tfidf(X_test, text_columns, prefix)
    elif text_representation == 'bert':
        label_encoder = load_encoder(f'le_bert_{section}.pkl')
        model = get_saved_model(len(label_encoder.classes_), section)
        text_representation_train = generate_bert(
            X_train, y_train, text_columns, model, label_encoder)
        text_representation_test = generate_bert(
            X_test, y_test, text_columns, model, label_encoder)

    X_train = hstack([
        *scaled_numeric_train,
        *categorical_dummies_train,
        *text_representation_train], format='csr')
    X_test = hstack([
        *scaled_numeric_test,
        *categorical_dummies_test,
        *text_representation_test], format='csr')
    return X_train, X_test


def encode_AD(
        X: DataFrame,
        numerical_columns: list,
        categorical_columns: list,
        text_columns: list,
        prefix: str) -> csr_matrix:
    X = X.reset_index(drop=True)

    if 'empenho_numero_do_processo' in categorical_columns:
        X = generate_fit_process_count(X)
        categorical_columns.remove('empenho_numero_do_processo')
        numerical_columns.append('empenhos_por_processo')

    scaled_numeric = generate_fit_scaler(X, numerical_columns, prefix)

    categorial_dummies = generate_fit_ohe(X, categorical_columns, prefix)

    text_representation = generate_fit_tfidf(X, text_columns, prefix)

    return hstack([*scaled_numeric, *categorial_dummies, *text_representation], format='csr')


def encode_inference(
        X: DataFrame,
        y: Series,
        numerical_columns: list,
        categorical_columns: list,
        text_columns: list,
        prefix: str,
        text_representation: str,
        section: str = None) -> csr_matrix:
    X = X.reset_index(drop=True)

    if 'empenho_numero_do_processo' in categorical_columns:
        X = generate_process_count(X)
        categorical_columns.remove('empenho_numero_do_processo')
        numerical_columns.append('empenhos_por_processo')

    scaled_numeric = generate_scaler(X, numerical_columns, prefix)

    categorical_dummies = generate_ohe(X, categorical_columns, prefix)

    if text_representation == 'tfidf':
        text_representation = generate_tfidf(X, text_columns, prefix)
    elif text_representation == 'bert':
        label_encoder = load_encoder(f'le_bert_{section}.pkl')
        model = get_saved_model(len(label_encoder.classes_), section)
        text_representation = generate_bert(
            X, y, text_columns, model, label_encoder)

    return hstack([*scaled_numeric, *categorical_dummies, *text_representation], format='csr')
