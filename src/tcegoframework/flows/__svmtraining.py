import gc
import warnings
from typing import Union

from pandas.core.frame import DataFrame

import tcegoframework.config as config
from scipy.sparse import csr_matrix, data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tcegoframework.data.filter import (change_scope, change_scope_dict, get_class92_mask, get_where_expired_class, get_where_under_threshold, get_where_zero_value, initialize_class_dict, create_scope_dict, masked_filter, min_docs_class,
                                        remove_class_92, remove_expired_classes,
                                        remove_zeroed_documents)
from tcegoframework.data.preprocessing.classification import (
    pp_second_tabular_training, pp_svm_training)
from tcegoframework.data.text import regularize_columns_name
from tcegoframework.dremio import get_train_data
from tcegoframework.io import (change_root_dir_name, dump_model, load_csv_data,
                               load_excel_data, save_scope_dict)
from tcegoframework.model.metrics import (classification_report_csv,
                                          special_report_csv)

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


def get_label_population_floor() -> int:
    return config.PARSER.getint(
        'options.training',
        'min_documents_class',
        fallback=2
    )


def dataframe_reset_index(dataframe: DataFrame) -> DataFrame:
    return dataframe.reset_index(drop=True)


def train():
    config.CHANGE_ROOT_DIR('TRAINING')

    if dataset_path := get_dataset_path():
        data = load_csv_data(dataset_path)
    else:
        data = get_train_data()

    if samples := get_sampling_number():
        data = data.sample(samples, random_state=config.RANDOM_SEED)

    data = dataframe_reset_index(data)
    data = regularize_columns_name(data)

    scope_dict = create_scope_dict(data)

    zero_value_mask = get_where_zero_value(data)
    data, _ = masked_filter(data, zero_value_mask)

    class92_mask = get_class92_mask(data)
    data, dropped_rows = masked_filter(data, class92_mask)
    scope_dict = change_scope_dict(
        scope_dict, dropped_rows['natureza_despesa_cod'], 'Classe 92')

    expired_classes = load_excel_data(path=get_expired_labels_path())
    expired_classes = regularize_columns_name(expired_classes)
    expired_classes_mask = get_where_expired_class(data, expired_classes)
    data, dropped_rows = masked_filter(data, expired_classes_mask)
    scope_dict = change_scope_dict(
        scope_dict, dropped_rows['natureza_despesa_cod'], 'Classe com vigência expirada')

    threshold = get_label_population_floor()
    under_threshold_mask = get_where_under_threshold(data, threshold)
    data, dropped_rows = masked_filter(data, under_threshold_mask)

    data = dataframe_reset_index(data)

    # MUDAR SALVAMENTO DO DICT E ATUALIZAR O FLUXO
    save_scope_dict('scope.pkl')

    # SVM Pipeline
    X_train, X_test, y_train, y_test = pp_svm_training(data.copy())
    model = SVC(kernel='linear', random_state=config.RANDOM_SEED)
    grid = {'C': [0.1, 1, 10, 50]}
    gs = GridSearchCV(model, grid, n_jobs=None, cv=3, verbose=10)
    gs.fit(csr_matrix(X_train.values), y_train)

    model = SVC(C=gs.best_params_['C'], kernel='linear',
                random_state=config.RANDOM_SEED)
    model.fit(csr_matrix(X_train.values), y_train)
    y_pred = model.predict(csr_matrix(X_test.values))
    special_report_csv(
        y_true=y_test,
        y_pred=y_pred,
        data=data,
        encoding=False,
        filename='svm_sp_rep.csv')
    dump_model(model=model, filename='svm_model.pkl')
    del model, X_train, X_test, y_train, y_test, y_pred, data
    gc.collect()

    # Modelo 2
    file = config.PARSER.get(
        'options.training',
        'validated_data_path',
        fallback=None)
    if file:
        data = load_excel_data(file)
        del file

    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)

    X_train, X_test, y_train, y_test = pp_second_tabular_training(data.copy())

    model = RandomForestClassifier(
        n_estimators=400, random_state=15, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    classification_report_csv(y_test, y_pred, False, 'rfii_clf_rep.csv')

    dump_model(model=model, filename='random_forest_ii_model.pkl')

    change_root_dir_name('PRODUCTION/')
