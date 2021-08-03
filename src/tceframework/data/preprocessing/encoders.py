from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tceframework.io import dump_encoder, load_encoder
from tceframework.data.text import clean_tfidf
from pandas import DataFrame


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
