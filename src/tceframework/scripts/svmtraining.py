import gc
import warnings

import tceframework.config as config
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tceframework.data.filter import (initialize_class_dict, min_docs_class,
                                      remove_class_92, remove_expired_classes,
                                      remove_zeroed_documents)
from tceframework.data.preprocessing.classification import (
    pp_second_tabular_training, pp_svm_training)
from tceframework.data.text import regularize_columns_name
from tceframework.dremio import get_train_data
from tceframework.io import (change_root_dir_name, dump_model, load_csv_data,
                             load_excel_data, save_scope_dict)
from tceframework.model.metrics import (classification_report_csv,
                                        special_report_csv)

warnings.filterwarnings('ignore')


def train():
    config.CHANGE_ROOT_DIR('TRAINING')

    # Loading data files and filtering

    file = config.PARSER.get(
        'options.training',
        'dataset_path',
        fallback=None)
    if file:
        data = load_csv_data(file)
        del file
    else:
        data = get_train_data()

    sample = config.PARSER.getint(
        'options.training',
        'sample_dataset',
        fallback=None)
    if sample:
        data = data.sample(sample, random_state=config.RANDOM_SEED)
        del sample

    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)

    initialize_class_dict(data)
    data = remove_zeroed_documents(data)
    data = remove_class_92(data)
    expired_data = load_excel_data(path=config.PARSER.get(
        'options.training',
        'expired_class_path'
    ))

    expired_data = regularize_columns_name(expired_data)
    data = remove_expired_classes(
        data=data,
        expired_data=expired_data
    )
    del expired_data
    data = min_docs_class(
        data=data,
        column='natureza_despesa_cod',
        threshold=config.PARSER.getint(
            'options.training',
            'min_documents_class',
            fallback=2
        )
    )

    save_scope_dict('scope.pkl')

    # n_classes = get_n_classes(data, 'natureza_despesa_cod')

    # SVM Pipeline

    X_train, X_test, y_train, y_test = pp_svm_training(data.copy())
    model = SVC(kernel='linear', random_state=config.RANDOM_SEED)
    grid = {'C': [0.1, 1, 10, 50]}
    gs = GridSearchCV(model, grid, n_jobs=-1, cv=3)
    gs.fit(X_train, y_train)

    model = SVC(C=gs.best_params_['C'], kernel='linear',
                random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # classification_report_csv(y_test, y_pred, False, 'rf_clf_rep.csv')
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
    # special_report_csv(
    #     y_true=y_test,
    #     y_pred=y_pred,
    #     data=data,
    #     encoding=False,
    #     filename='rf_sp_rep.csv')
    dump_model(model=model, filename='random_forest_ii_model.pkl')

    change_root_dir_name('PRODUCTION/')
