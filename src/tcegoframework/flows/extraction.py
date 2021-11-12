
import warnings

from pandas.core.frame import DataFrame
from tcegoframework.dremio import construct_query, execute_query
from tcegoframework.preprocessing.text import regularize_columns_name

warnings.filterwarnings('ignore')

def query_dataset(filters: dict) -> DataFrame:
    query = construct_query(filters)
    data = execute_query(query)
    return data


def extraction_flow(filters: dict):
    print('Preparando e executando consulta...')
    data = query_dataset(filters)
    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)
    print('Salvando base de dados...')
    data.to_csv('tcedata.csv', index=False)
    print('Finalizado.')
