# %%
import time
import warnings
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (explained_variance_score, mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from preprocessing.dataprep import (data_preparation, encode_train_test,
                                    filter_tce_data)
from preprocessing.text import fixColumnName

warnings.filterwarnings('ignore')

# %%
start_time = time.time()
start_date = datetime.now()
print(f'Starting evaluation at {start_date.strftime("%d/%m/%Y %H:%M:%S")}')
print()
print('Loading data...')

data_loading_time = time.time()
data = pd.read_csv('../database/dadosTCE.csv',
                   low_memory=False, encoding='utf-8')
print(f'Data loading time: {(time.time() - data_loading_time):.2f}s')
print()

data_prep_time = time.time()
print('Preprocessing data...')

# data = data.sample(1000,).reset_index(drop=True)
data.columns = list(map(fixColumnName, data.columns))
data, _ = filter_tce_data(data, '../database/norel.xlsx')

categorical_columns = [
    # 'exercicio_do_orcamento_ano',
    # 'empenho_sequencial_empenho',
    'orgao',
    'orgao_sucessor_atual',
    'tipo_administracao_nome',
    'tipo_poder_nome',
    # 'classificacao_orcamentaria_descricao',
    'funcao',
    'subfuncao',
    'programa',
    'acao',
    'grupo_despesa',
    'elemento_despesa',
    # 'natureza_despesa_cod',
    # 'natureza_despesa_nome',
    # 'formalidade_nome',
    'modalidade_licitacao_nome',
    # 'fonte_recurso_cod',
    'fonte_recurso_nome',
    'beneficiario_cnpj',
    'beneficiario_cpf',
    'beneficiario_cpf/cnpj',
    # 'periodo',
    'empenho_numero_do_processo',
    # 'empenho_sequencial_empenho.1',
]

text_columns = [
    # 'beneficiario_nome',
    'empenho_historico',
]

numerical_columns = [
    # 'valor_empenhado',
    # 'valor_anulacao_empenho',
    # # 'valor_estorno_anulacao_empenho',
    # 'valor_cancelamento_empenho',
    # # 'valor_anulacao_cancelamento_empenho',
    # 'valor_saldo_do_empenho',
    # 'valor_liquidacao_empenho',
    # 'valor_anulacao_liquidacao_empenho',
    # 'valor_saldo_liquidado',
    # 'valor_ordem_de_pagamento',
    # 'valor_guia_recolhimento',
    # 'valor_anulacao_ordem_de_pagamento',
    # 'valor_estorno_anulacao_o._pagamento',
    # 'valor_estorno_guia_recolhimento',
    # 'valor_saldo_pago',
    # 'valor_saldo_a_pagar',
    # 'valor_a_liquidar',
    # 'valor_a_pagar_liquidado'
]

target = data.valor_empenhado

data = data.loc[:, (*categorical_columns,
                    *text_columns,
                    *numerical_columns,)]

data, categorical_columns, numerical_columns = data_preparation(
    data,
    target,
    test_size=0.3,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns,
    text_columns=text_columns,)

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=15)

X_train, X_test = encode_train_test(
    X_train, X_test, numerical_columns, categorical_columns, text_columns, tfidf=True,)

print(f'Data preparation time: {(time.time() - data_prep_time):.2f}s')
print(f'Training shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')
print()
# %%
svm_time = time.time()
print('Training SVR...')

reg = SVR(kernel='linear', C=10)
reg.fit(X_train, y_train)

print(f'SVM training time: {(time.time() - svm_time):.2f}s')
print()
# %%
svm_time = time.time()
y_pred = reg.predict(X_test)

print(f'SVM predict time: {(time.time() - svm_time):.2f}s')
print()
# %%
print(f'EVS: {explained_variance_score(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R^2: {r2_score(y_test, y_pred)}')
print()

# %%
rf_time = time.time()
print('Training RandomForest...')
reg = RandomForestRegressor(n_estimators=1500, random_state=15, n_jobs=-1)
reg.fit(X_train, y_train)

print(f'RF training time: {(time.time() - rf_time):.2f}s')
print()
# %%
rf_time = time.time()
y_pred = reg.predict(X_test)

print(f'RF trining time: {(time.time() - rf_time):.2f}s')
print()
# %%
print(f'EVS: {explained_variance_score(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R^2: {r2_score(y_test, y_pred)}')
print()
# %%
finish_date = datetime.now()
print(f'Finishing evaluation at {finish_date.strftime("%d/%m/%Y %H:%M:%S")}')
