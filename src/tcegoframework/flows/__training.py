import warnings

import tcegoframework.config as config
from tcegoframework.data.filter import (min_docs_class, remove_class_92,
                                        remove_expired_classes, initialize_class_dict,
                                        remove_zeroed_documents)

from tcegoframework.io import (change_root_dir_name, dump_model, load_csv_data,
                               load_encoder, load_excel_data, load_torch_model,
                               save_scope_dict,
                               )
from tcegoframework.dremio import get_train_data
from tcegoframework.model.bert import NaturezaClassifier, fit_bert, get_predictions
from tcegoframework.model.metrics import classification_report_csv, special_report_csv
from sklearn.ensemble import RandomForestClassifier
from torch.cuda import empty_cache
import gc

warnings.filterwarnings('ignore')


@DeprecationWarning
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

    n_classes = get_n_classes(data, 'natureza_despesa_cod')

    # Bert Pipeline
    train_data_loader, test_data_loader = pp_bert_training(data=data.copy())
    bert_model = NaturezaClassifier(n_classes=n_classes)
    _ = fit_bert(
        model=bert_model,
        epochs=config.PARSER.getint(
            'options.training',
            'epochs',
            fallback=5
        ),
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,)
    del bert_model, train_data_loader
    gc.collect()

    # Load BERT model
    bert_model = NaturezaClassifier(n_classes)
    bert_model = bert_model.to(config.BERT_DEVICE)
    state_dict = load_torch_model(filename='bert_model.bin')
    bert_model.load_state_dict(state_dict)
    del state_dict, n_classes
    gc.collect()

    # Bert Evaluation
    result = get_predictions(model=bert_model, data_loader=test_data_loader)
    del bert_model, test_data_loader
    gc.collect()

    # classification_report_csv(result['real_values'], result['predictions'], True, 'bert_clf_rep.csv')
    special_report_csv(
        y_true=result['real_values'],
        y_pred=result['predictions'],
        data=data,
        encoding=True,
        filename='bert_sp_rep.csv')

    del result

    X_train, X_test, y_train, y_test = pp_tabular_training(data.copy())
    empty_cache()
    gc.collect()

    model = RandomForestClassifier(
        n_estimators=700, random_state=15, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # classification_report_csv(y_test, y_pred, False, 'rf_clf_rep.csv')
    special_report_csv(
        y_true=y_test,
        y_pred=y_pred,
        data=data,
        encoding=False,
        filename='rf_sp_rep.csv')
    dump_model(model=model, filename='random_forest_model.pkl')
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
