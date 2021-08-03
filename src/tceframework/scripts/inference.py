import warnings

import tceframework.config as config
from tceframework.data.filter import scope_filter
from tceframework.data.inference.utils import initialize_inference_dict, resolve_output
from tceframework.data.preprocessing.classification import pp_second_tabular_inference, pp_tabular_inference
from tceframework.data.text import regularize_columns_name
from tceframework.dremio import construct_query, execute_query
from tceframework.io import load_model, save_inference_results, save_json
from datetime import datetime

warnings.filterwarnings('ignore')


def inference(filters: dict):
    config.CHANGE_ROOT_DIR('PRODUCTION')

    # Executing query
    query = construct_query(filters)
    data = execute_query(query)
    del query
    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)

    initialize_inference_dict(data)

    data = scope_filter(data)

    X = pp_tabular_inference(data.copy())
    model = load_model(filename='random_forest_model.pkl')
    y_pred_class = model.predict(X)
    del model, X

    X = pp_second_tabular_inference(data.copy())
    model = load_model(filename='random_forest_ii_model.pkl')
    y_pred_correctness = model.predict(X)
    del model, X

    resolve_output(data, y_pred_class, y_pred_correctness)

    date = datetime.today().strftime('%d-%m-%Y')

    save_inference_results(f'{date}_results.csv')
    save_json(filters, f'{date}_applied_filters.json')
