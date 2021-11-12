
import warnings

from tcegoframework.dremio import get_train_data
from tcegoframework.preprocessing.text import regularize_columns_name

warnings.filterwarnings('ignore')

def extraction_flow(filters: dict):
    print('Preparando e executando consulta...')
    data = get_train_data()
    data = data.reset_index(drop=True)
    data = regularize_columns_name(data)
    print('Salvando base de dados...')
    data.to_csv('tcedata.csv', index=False)
    print('Finalizado.')
