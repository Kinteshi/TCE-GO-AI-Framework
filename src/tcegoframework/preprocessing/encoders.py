from numpy import zeros
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tcegoframework import config
from tcegoframework.data.misc import create_data_loader
from tcegoframework.io import dump_encoder, load_encoder
from pandas import DataFrame
from tcegoframework.model.bert import get_saved_model, generate_bert_representation
from transformers import BertTokenizer

from tcegoframework.preprocessing.text import clean_nlp, clean_tfidf


def generate_fit_scaler(X_train, columns, prefix=''):
    for col_name in columns:
        scaler = MinMaxScaler()
        X_train[col_name].update(scaler.fit_transform(
            X_train[col_name].values.reshape(-1, 1)).flatten())
        dump_encoder(scaler, f'{prefix}_sc_{col_name}.pkl')
        del scaler
    return X_train


def generate_scaler(X_test, columns, prefix=''):
    for col_name in columns:
        scaler = load_encoder(f'{prefix}_sc_{col_name}.pkl')
        X_test[col_name].update(scaler.transform(
            X_test[col_name].values.reshape(-1, 1)).flatten())
    return X_test


def generate_fit_ohe(X_train, columns, prefix=''):
    for col_name in columns:
        ohe = OneHotEncoder(handle_unknown='ignore')
        enc = ohe.fit_transform(X_train[col_name].values.reshape(-1, 1))
        enc = DataFrame.sparse.from_spmatrix(
            enc, columns=ohe.get_feature_names(input_features=(col_name,)))
        X_train.drop([col_name], inplace=True, axis='columns')
        X_train = X_train.join(enc)
        dump_encoder(ohe, f'{prefix}_ohe_{col_name}.pkl')
        del ohe
    return X_train


def generate_ohe(X_test, columns, prefix=''):
    for col_name in columns:
        ohe = load_encoder(f'{prefix}_ohe_{col_name}.pkl')
        enc = ohe.transform(X_test[col_name].values.reshape(-1, 1))
        enc = DataFrame.sparse.from_spmatrix(
            enc, columns=ohe.get_feature_names(input_features=(col_name,)))
        X_test.drop([col_name], inplace=True, axis='columns')
        X_test = X_test.join(enc)
    return X_test


def generate_fit_tfidf(X_train, columns, prefix=''):
    for col_name in columns:
        X_train[col_name].update(X_train[col_name].map(clean_tfidf))
        tfv = TfidfVectorizer()
        tfidf = tfv.fit_transform(X_train[col_name])
        columns_names = ['Tfidf_' +
                         word for word in tfv.get_feature_names()]
        tfidf = DataFrame.sparse.from_spmatrix(tfidf, columns=columns_names)
        X_train.drop([col_name], inplace=True, axis='columns')
        X_train = X_train.join(tfidf)
        dump_encoder(tfv, f'{prefix}_tfv_{col_name}.pkl')
        del tfv
    return X_train


def generate_tfidf(X_test, columns, prefix=''):
    for col_name in columns:
        X_test[col_name].update(X_test[col_name].map(clean_tfidf))
        tfv = load_encoder(f'{prefix}_tfv_{col_name}.pkl')
        tfidf = tfv.transform(X_test[col_name])
        columns_names = ['Tfidf_' +
                         word for word in tfv.get_feature_names()]
        tfidf = DataFrame.sparse.from_spmatrix(tfidf, columns=columns_names)
        X_test.drop([col_name], inplace=True, axis='columns')
        X_test = X_test.join(tfidf)
    return X_test


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
        representation = generate_bert_representation(model, data_loader)
        X.drop([col_name], inplace=True, axis='columns')
        X = X.join(representation)
    return X


def encode_train_test(X_train: DataFrame, X_test: DataFrame, numerical_columns: list, categorical_columns: list,
                      text_columns: list, y_train, y_test, prefix: str, text_representation: str) -> DataFrame:
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

    if text_representation == 'tfidf':
        X_train = generate_fit_tfidf(X_train, text_columns, prefix)
        X_test = generate_tfidf(X_test, text_columns, prefix)
    elif text_representation == 'bert':
        label_encoder = load_encoder('targetNLP.pkl')
        model = get_saved_model(len(label_encoder.classes_))
        X_train = generate_bert_representation(
            X_train, y_train, text_columns, model, label_encoder)
        X_test = generate_bert_representation(
            X_test, y_test, text_columns, model, label_encoder)

    return X_train, X_test


def encode_inference(X, y, numerical_columns: list, categorical_columns: list, text_columns: list, prefix: str,
                     text_representation: str) -> DataFrame:
    X = X.reset_index(drop=True)

    if 'empenho_numero_do_processo' in categorical_columns:
        X = generate_process_count(X)
        categorical_columns.remove('empenho_numero_do_processo')
        numerical_columns.append('empenhos_por_processo')

    X = generate_scaler(X, numerical_columns, prefix)

    X = generate_ohe(X, categorical_columns, prefix)

    if text_representation == 'tfidf':
        X = generate_tfidf(X, text_columns, prefix)
    elif text_representation == 'bert':
        label_encoder = load_encoder('targetNLP.pkl')
        model = get_saved_model(len(label_encoder.classes_))
        X = generate_bert_representation(
            X, y, text_columns, model, label_encoder)

    return X
