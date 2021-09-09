from datetime import datetime

from dateutil.parser import parse
from numpy import array, zeros
from pandas import DataFrame


def get_n_classes(data: DataFrame, column: str) -> int:
    n_classes = data[column].value_counts().shape[0]
    return n_classes


def code_reaper(input_text):
    return input_text[7:]


def date_to_month(input_date):
    if isinstance(input_date, int):
        date = datetime.fromordinal(
            datetime(1900, 1, 1).toordinal() +
            input_date -
            2)
    else:
        date = parse(input_date)
    return date.month


def data_preparation(data: DataFrame, categorical_columns, numerical_columns):

    data = data.reset_index(drop=True)
    # Criação de meta-atributos e tratamento comum para todos os dados

    # Criação do meta-atributo mês
    if 'periodo' in categorical_columns:
        data['periodo'].update(data['periodo'].map(date_to_month))

    # Criação do meta-atributo "Pessoa Jurídica?"
    pessoa_juridica = array(
        [1 if cpf == '-' else 0
         for cpf in data['beneficiario_cpf'].values])
    data['pessoa_juridica'] = pessoa_juridica.astype('int8')

    data.drop(['beneficiario_cpf', 'beneficiario_cnpj',
               'beneficiario_cpf/cnpj'], axis='columns', inplace=True)
    categorical_columns.remove('beneficiario_cpf')
    categorical_columns.remove('beneficiario_cnpj')
    categorical_columns.remove('beneficiario_cpf/cnpj')
    categorical_columns.append('pessoa_juridica')
    del pessoa_juridica

    # Codigo que gera o meta atributo "orgao_sucedido" onde 1 representa que o orgao tem um novo orgao sucessor e 0 caso contrario
    orgao_sucedido = zeros(data.shape[0])

    for i in range(data.shape[0]):
        if(data['orgao'].iloc[i] != data['orgao_sucessor_atual'].iloc[i]):
            orgao_sucedido[i] = 1

    data['orgao_sucedido'] = orgao_sucedido.astype('int8')
    data.drop(['orgao'], axis='columns', inplace=True)
    categorical_columns.remove('orgao')
    categorical_columns.append('orgao_sucedido')
    del orgao_sucedido

    # Codigo que retira o codigo de programa (retirando 10 valores)
    data['programa'] = data['programa'].map(code_reaper)
    # Codigo que retira o codigo de acao (retirando 77 valores)
    data['acao'] = data['acao'].map(code_reaper)

    # Codigo que concatena acao e programa
    acao_programa = zeros(data.shape[0], dtype='object')
    for i in range(data.shape[0]):
        acao_programa[i] = (data['acao'].iloc[i] + ' & ' +
                            data['programa'].iloc[i])
    data['acao_programa'] = acao_programa
    data.drop(['acao', 'programa'], axis='columns', inplace=True)
    categorical_columns.remove('acao')
    categorical_columns.remove('programa')
    categorical_columns.append('acao_programa')
    del acao_programa

    return data, categorical_columns, numerical_columns