from functools import partial
import gc
import time
import warnings
from typing import Union

from pandas.core.frame import DataFrame
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import tcegoframework.config as config
from tcegoframework.data.filter import (
    change_scope_dict, create_scope_dict, masked_filter, where_below_threshold, where_class_92, where_expired_class, where_zero_value)
from tcegoframework.dremio import construct_query, execute_query, get_train_data
from tcegoframework.io import (dump_model, load_csv_data, load_excel_data,
                               save_scope_dict)
from tcegoframework.model.metrics import (classification_report_csv,
                                          special_report_csv)
from tcegoframework.preprocessing.classification import preprocessing_training_corretude, preprocessing_training_natureza
from tcegoframework.preprocessing.text import regularize_columns_name

warnings.filterwarnings('ignore')


def get_dataset_path() -> Union[str, None]:
    return config.PARSER.get(
        'options.training',
        'dataset_path',
        fallback=None)


def get_sampling_number() -> Union[int, None]:
    return config.PARSER.getint(
        'options.training',
        'sample_dataset',
        fallback=None)


def get_expired_labels_path() -> str:
    return config.PARSER.get(
        'options.training',
        'expired_class_path'
    )


def get_validated_data_path() -> str:
    file = config.PARSER.get(
        'options.training',
        'validated_data_path',)
    return file


def get_label_population_floor() -> int:
    return config.PARSER.getint(
        'options.training',
        'min_documents_class',
        fallback=2
    )


def get_algorithm() -> str:
    return config.PARSER.get(
        'options.training',
        'algorithm',
        fallback='svm'
    )


def dataframe_reset_index(dataframe: DataFrame) -> DataFrame:
    return dataframe.reset_index(drop=True)


def data_cleaning(data: DataFrame) -> tuple[DataFrame, DataFrame]:
    data = regularize_columns_name(data)

    print(f'Dimensões iniciais: {data.shape}')
    print(f'Quantidade de classes: {data.natureza_despesa_cod.unique().shape}')
    print()

    scope_dict = create_scope_dict(data)

    zero_value_mask = where_zero_value(data)
    data, dropped_rows = masked_filter(data, zero_value_mask)
    print()
    print(
        f'Dimensões após remoção de empenhos com saldo zerados: {data.shape}')
    print(
        f'Quantidade de classes restantes: {data.natureza_despesa_cod.unique().shape}')
    print(f'Dimensões dos empenhos removidos: {dropped_rows.shape}')
    print(
        f'Quantidade de classes removidas: {dropped_rows.natureza_despesa_cod.unique().shape}')
    print()

    class92_mask = where_class_92(data)
    data, dropped_rows = masked_filter(data, class92_mask)
    scope_dict = change_scope_dict(
        scope_dict, dropped_rows['natureza_despesa_cod'], 'Classe 92')
    print()
    print(f'Dimensões após remoção de empenhos da Classe 92: {data.shape}')
    print(
        f'Quantidade de classes restantes: {data.natureza_despesa_cod.unique().shape}')
    print(f'Dimensões dos empenhos removidos: {dropped_rows.shape}')
    print(
        f'Quantidade de classes removidas: {dropped_rows.natureza_despesa_cod.unique().shape}')
    print()

    expired_classes = load_excel_data(path=get_expired_labels_path())
    expired_classes = regularize_columns_name(expired_classes)
    expired_classes_mask = where_expired_class(data, expired_classes)
    data, dropped_rows = masked_filter(data, expired_classes_mask)
    scope_dict = change_scope_dict(
        scope_dict, dropped_rows['natureza_despesa_cod'], 'Classe com vigência expirada')
    print()
    print(
        f'Dimensões após remoção de empenhos com vigência encerrada: {data.shape}')
    print(
        f'Quantidade de classes restantes: {data.natureza_despesa_cod.unique().shape}')
    print(f'Dimensões dos empenhos removidos: {dropped_rows.shape}')
    print(
        f'Quantidade de classes removidas: {dropped_rows.natureza_despesa_cod.unique().shape}')
    print()

    threshold = get_label_population_floor()
    under_threshold_mask = where_below_threshold(data, threshold)
    data_above, data_below = masked_filter(data, under_threshold_mask)
    data_below, _ = masked_filter(
        data_below, where_below_threshold(data_below, 2))
    print()
    print(
        f'Dimensões após separação dos empenhos de classes pequenas: {data_above.shape}')
    print(
        f'Quantidade de classes restantes: {data_above.natureza_despesa_cod.unique().shape}')
    print(f'Dimensões dos empenhos de classes pequenas: {data_below.shape}')
    print(
        f'Quantidade de classes pequenas: {data_below.natureza_despesa_cod.unique().shape}')
    print()

    save_scope_dict('scope.pkl', scope_dict)

    return data_above, data_below


def get_dataset(filters: dict) -> DataFrame:
    if dataset_path := get_dataset_path():
        data = load_csv_data(dataset_path)
    else:
        if filters:
            query = construct_query(filters)
            data = execute_query(query)
        else:
            data = get_train_data()
    return data


def model_name(type: str, algorithm: str):
    return f'{type}_{algorithm}'


def train_rf_natureza(data: DataFrame, section: str):
    X_train, X_test, y_train, y_test = preprocessing_training_natureza(
        data.copy(), 'tfidf', section)

    model = RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    special_report_csv(
        y_true=y_test,
        y_pred=y_pred,
        data=data,
        encoding=False,
        filename=f'rf_natureza_{section}_rep.csv')
    dump_model(model=model, filename=f'rf_natureza_{section}_model.pkl')


def train_svm_natureza(data: DataFrame, section: str):
    X_train, X_test, y_train, y_test = preprocessing_training_natureza(
        data.copy(), 'tfidf', section)

    model = SVC(kernel='linear', random_state=config.RANDOM_SEED)
    grid = {'C': [0.1, 1, 10, 50]}
    gs = GridSearchCV(model, grid, n_jobs=None, cv=3, verbose=10)
    gs.fit(csr_matrix(X_train.values), y_train)

    model = SVC(C=gs.best_params_['C'], kernel='linear',
                random_state=config.RANDOM_SEED, probability=True)
    model.fit(csr_matrix(X_train.values), y_train)
    y_pred = model.predict(csr_matrix(X_test.values))

    special_report_csv(
        y_true=y_test,
        y_pred=y_pred,
        data=data,
        encoding=False,
        filename=f'svm_natureza_{section}_rep.csv')
    dump_model(model=model, filename=f'svm_natureza_{section}_model.pkl')


def train_corretude(data: DataFrame):
    X_train, X_test, y_train, y_test = preprocessing_training_corretude(
        data.copy())

    model = RandomForestClassifier(
        n_estimators=400, random_state=15, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    classification_report_csv(y_test, y_pred, False, 'rf_corretude_rep.csv')

    dump_model(model=model, filename='rf_corretude_model.pkl')


def train_flow(filters: dict):
    print('Carregando base de dados...')
    data = get_dataset(filters)
    print('Base de dados carregada.')

    if samples := get_sampling_number():
        data = data.sample(samples, random_state=config.RANDOM_SEED)
        data = dataframe_reset_index(data.copy())

    print(f'Iniciando limpeza...')
    data_above, data_below = data_cleaning(data.copy())
    print('Limpeza finalizada')

    # BORDA DE TRANSIÇÃO ENTRE LIMPEZA E INÍCIO DO PREPROCESSAMENTO

    if get_algorithm() == 'svm':
        train_natureza = partial(train_svm_natureza)
    elif get_algorithm() == 'rf':
        train_natureza = partial(train_rf_natureza)

    print('Iniciando treinamento do classificador de natureza de despesa')
    print('Treinamento do modelo de classes maiores')
    time_ref = time.time()
    train_natureza(data_above.copy(), 'above')
    print(
        f'Treinamento finalizado. Duração total: {(time.time() - time_ref)/60}')
    print('Treinamento do modelo de classes menores')
    time_ref = time.time()
    train_natureza(data_below.copy(), 'below')
    print(
        f'Treinamento finalizado. Duração total: {(time.time() - time_ref)/60}')

    # Modelo 2

    data = load_excel_data(get_validated_data_path())
    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)
    print('Iniciando treinamento do classificador de corretude')
    time_ref = time.time()
    train_corretude(data.copy())
    print(
        f'Treinamento finalizado. Duração total: {(time.time() - time_ref)/60}')
    print('Script de treinamento finalizado')
