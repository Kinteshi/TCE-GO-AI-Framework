import jaydebeapi
from pandas import DataFrame
from datetime import date, timedelta
import tcegoframework.config as config
from pkg_resources import resource_filename


# Variuaveis
# Caminho para driver JDBC do Dremio.
# pathDremioJDBC = 'dremio-jdbc-driver-14.0.0-202103011714040666-9a0c2e10.jar'
path_dremio_driver = resource_filename(
    'tcegoframework.resources', 'dremio-jdbc-driver.jar')

# Consulta Base Dremio - Retorna dados do ultimo dia de recepção dos dados
base_query = 'SELECT "Exercício do orçamento (Ano)" ' \
    ' ,"Órgão (Código/Nome)"' \
    ' ,"Órgão Sucessor Atual (Código/Nome)"' \
    ' ,"Tipo Administração (Nome)"' \
    ' ,"Tipo Poder (Nome)"' \
    ' ,"Classificação orçamentária (Descrição)"' \
    ' ,"Função (Cod/Nome)"' \
    ' ,"Subfunção (Cod/Nome)"' \
    ' ,"Programa (Cod/Nome)"' \
    ' ,"Ação (Cod/Nome)"' \
    ' ,"Natureza Despesa (Cod)"' \
    ' ,"Natureza Despesa (Nome)"' \
    ' ,"Grupo Despesa (Cod/Nome)"' \
    ' ,"Elemento Despesa (Cod/Nome)"' \
    ' ,"Formalidade (Nome)"' \
    ' ,"Modalidade Licitação (Nome)"' \
    ' ,"Fonte Recurso (Cod)"' \
    ' ,"Fonte Recurso (Nome)"' \
    ' ,"Beneficiário (Nome)"' \
    ' ,"Beneficiário (CPF)"' \
    ' ,"Beneficiário (CNPJ)"' \
    ' ,"Beneficiário (CPF/CNPJ)"' \
    ' ,"Período (Dia/Mes/Ano)"' \
    ' ,"Empenho (Sequencial Empenho)"' \
    ' ,"Empenho (Histórico)"' \
    ' ,"Empenho (Número do Processo)"' \
    ' ,"Valor Empenhado"' \
    ' ,"Valor Anulação Empenho"' \
    ' ,"Valor Estorno Anulação Empenho"' \
    ' ,"Valor Cancelamento Empenho"' \
    ' ,"Valor Anulação Cancelamento Empenho"' \
    ' ,"Valor Liquidação Empenho"' \
    ' ,"Valor Anulação Liquidacao Empenho"' \
    ' ,"Valor Ordem de Pagamento"' \
    ' ,"Valor Guia Recolhimento"' \
    ' ,"Valor Anulação Ordem de Pagamento"' \
    ' ,"Valor Estorno Anulação O. Pagamento"' \
    ' ,"Valor Estorno Guia Recolhimento"' \
    ' ,"Valor Saldo do Empenho"' \
    ' ,"Valor Saldo Liquidado"' \
    ' ,"Valor Saldo Pago"' \
    ' ,"Valor Saldo a Pagar"' \
    ' ,"Valor a Liquidar"' \
    ' ,"Valor a Pagar Liquidado" ' \
    ' FROM IE.EXTRACAO.EOFClassificacao c ' \



def get_connection():
    conn = jaydebeapi.connect(
        'com.dremio.jdbc.Driver',
        config.PARSER.get(
            'options.dremio',
            'connection'),
        [
            config.PARSER.get(
                'options.dremio',
                'user'
            ),
            config.PARSER.get(
                'options.dremio',
                'password'
            )
        ],
        path_dremio_driver
    )

    return conn


def execute_query(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    data = DataFrame(
        cursor.fetchall(),
        columns=[i[0] for i in cursor.description]
    )
    conn.close()
    return data


def get_train_data(filters: dict):
    query = base_query + ' WHERE c."Exercício do orçamento (Ano)" >= 2015'
    data = execute_query(query)
    return data


def construct_query(filters):
    # dates daterange orgaos

    conditions = []

    if 'dates' in filters:
        for day in filters['dates']:
            day = day.strftime('%Y/%m/%d')
            condition = f'c."Período (Dia/Mes/Ano)" = \'{day}\''
            conditions.append(condition)
    elif 'daterange' in filters:
        date_a, date_b = filters['daterange']
        if date_a > date_b:
            # p < a and p > b
            date_a = date_a.strftime('%Y/%m/%d')
            condition = f'c."Período (Dia/Mes/Ano)" <= \'{date_a}\''
            conditions.append(condition)
            date_b = date_b.strftime('%Y/%m/%d')
            condition = f'c."Período (Dia/Mes/Ano)" >= \'{date_b}\''
            conditions.append(condition)
        else:
            date_a = date_a.strftime('%Y/%m/%d')
            condition = f'c."Período (Dia/Mes/Ano)" >= \'{date_a}\''
            conditions.append(condition)
            date_b = date_b.strftime('%Y/%m/%d')
            condition = f'c."Período (Dia/Mes/Ano)" <= \'{date_b}\''
            conditions.append(condition)
    else:
        # DIA ANTERIOR
        previous_day = (date.today() - timedelta(days=1)).strftime('%Y/%m/%d')
        condition = f'c."Período (Dia/Mes/Ano)" = \'{previous_day}\''
        conditions.append(condition)
    if 'orgaos' in filters:
        for orgao in filters['orgaos']:
            condition = f'c."Órgão (Código/Nome)" = {orgao}'
            conditions.append(condition)

    query = base_query + ' WHERE ' + ' and '.join(conditions)

    return query
