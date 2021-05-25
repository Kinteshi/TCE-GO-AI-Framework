import copy
import nltk
import numpy as np
import pandas as pd
from cleantext import clean
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pandas.core.arrays import categorical
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Union

from preprocessing.text import fixColumnName
from preprocessing.classification import tratarLabel

global_categorical_columns = [
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
    'formalidade_nome',
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

global_text_columns = [
    # 'beneficiario_nome',
    'empenho_historico',
]

global_numerical_columns = [
    # 'valor_empenhado',
    # 'valor_anulacao_empenho',
    # 'valor_estorno_anulacao_empenho',
    # 'valor_cancelamento_empenho',
    # 'valor_anulacao_cancelamento_empenho',
    'valor_saldo_do_empenho',
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


def stemming(input_text):
    porter = PorterStemmer()
    words = input_text.split()
    stemmed_words = [porter.stem(word) for word in words]
    return " ".join(stemmed_words)


def remove_stopwords(input_text, stopwords_list):
    words = input_text.split()
    clean_words = [word for word in words if (
        word not in stopwords_list) and len(word) > 1]
    return " ".join(clean_words)


def text_preprocessing(input_text):

    text = clean(
        input_text,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        normalize_whitespace=True,
        no_line_breaks=True,
        strip_lines=True,
        keep_two_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=True,
        no_digits=True,
        no_currency_symbols=True,
        no_punct=True,
        no_emoji=True,
        replace_with_url="url",
        replace_with_email="email",
        replace_with_phone_number="telefone",
        replace_with_number="",
        replace_with_digit="",
        replace_with_currency_symbol="BRL",
        replace_with_punct=" ",
        lang="pt",
    )

    try:
        stopwords_list = stopwords.words('portuguese')
    except:
        nltk.download("stopwords")
        nltk.download('wordnet')
        stopwords_list = stopwords.words('portuguese')

    stopwords_list.append("pdf")
    stopwords_list.append("total")
    stopwords_list.append("mes")
    stopwords_list.append("goias")
    stopwords_list.append("go")
    text = remove_stopwords(text, stopwords_list)
    text = stemming(text)
    return text


def generate_fit_scaler(X_train, columns):
    scaler_dict = {}
    for col_name in columns:
        scaler = MinMaxScaler()
        X_train[col_name] = scaler.fit_transform(
            X_train[col_name].values.reshape(-1, 1))
        scaler_dict[col_name] = copy.deepcopy(scaler)
        del scaler
    return X_train, scaler_dict


def generate_scaler(X_test, columns, scaler_dict):
    for col_name in columns:
        scaler = scaler_dict[col_name]
        X_test[col_name] = scaler.transform(
            X_test[col_name].values.reshape(-1, 1))
    return X_test


def generate_fit_ohe(X_train, columns):
    ohe_dict = {}
    for col_name in columns:
        ohe = OneHotEncoder(handle_unknown='ignore')
        enc = ohe.fit_transform(X_train[col_name].values.reshape(-1, 1))
        enc = pd.DataFrame(enc.toarray(), columns=ohe.get_feature_names(
            input_features=(col_name,)))
        X_train.drop([col_name], inplace=True, axis='columns')
        X_train = X_train.join(enc)
        ohe_dict[col_name] = copy.deepcopy(ohe)
        del ohe
    return X_train, ohe_dict


def generate_ohe(X_test, columns, ohe_dict):
    for col_name in columns:
        ohe = ohe_dict[col_name]
        enc = ohe.transform(X_test[col_name].values.reshape(-1, 1))
        enc = pd.DataFrame(enc.toarray(), columns=ohe.get_feature_names(
            input_features=(col_name,)))
        X_test.drop([col_name], inplace=True, axis='columns')
        X_test = X_test.join(enc)
    return X_test


def generate_fit_tfidf(X_train, columns):
    tfv_dict = {}
    for col_name in columns:
        X_train[col_name] = X_train[col_name].map(
            text_preprocessing)
        tfv = TfidfVectorizer()
        tfidf = tfv.fit_transform(X_train[col_name])
        columns_names = ['Tfidf_' +
                         word for word in tfv.get_feature_names()]
        tfidf = pd.DataFrame(tfidf.toarray(), columns=columns_names)
        X_train.drop([col_name], inplace=True, axis='columns')
        X_train = X_train.join(tfidf)
        tfv_dict[col_name] = copy.deepcopy(tfv)
        del tfv
    return X_train, tfv_dict


def generate_tfidf(X_test, columns, tfv_dict):

    for col_name in columns:
        X_test[col_name] = X_test[col_name].map(
            text_preprocessing)
        tfv = tfv_dict[col_name]
        tfidf = tfv.transform(X_test[col_name])
        columns_names = ['Tfidf_' +
                         word for word in tfv.get_feature_names()]
        tfidf = pd.DataFrame(tfidf.toarray(), columns=columns_names)
        X_test.drop([col_name], inplace=True, axis='columns')
        X_test = X_test.join(tfidf)
    return X_test


def code_reaper(input_text):
    return input_text[7:]


def filter_tce_data(data, path_to_vigencie_encerrada):
    # Excluindo empenhos diferentes aglomerados na classe 92
    exercicio_anterior = data['natureza_despesa_cod'].str.contains(
        ".\..\...\.92\...", regex=True, na=False)
    index = exercicio_anterior.where(exercicio_anterior == True).dropna().index
    data.drop(index, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Deletando empenhos sem relevancia devido ao saldo zerado
    index = data['valor_saldo_do_empenho'].where(
        data['valor_saldo_do_empenho'] == 0).dropna().index
    data.drop(index, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Funcao que gera o rotulo e retorna as linhas com as naturezas de despesa que so aparecem em 1 empenho
    target, linhas_label_unica = tratarLabel(data)
    target = pd.DataFrame(target)

    # Excluindo as naturezas de despesas que so tem 1 empenho
    data = data.drop(linhas_label_unica)
    data.reset_index(drop=True, inplace=True)
    del linhas_label_unica

    # Excluindo empenhos irrelevantes devido nao estarem mais em vigencia
    sem_relevancia = pd.read_excel(
        path_to_vigencie_encerrada)
    sem_relevancia = sem_relevancia['Nat. Despesa']
    sem_relevancia = pd.DataFrame(sem_relevancia)
    excluir = []
    for i in range(len(sem_relevancia['Nat. Despesa'])):
        excluir.append(
            target.where(
                target['natureza_despesa_cod'] == sem_relevancia['Nat. Despesa'].iloc[i]
            ).dropna().index
        )
    excluir = [item for sublist in excluir for item in sublist]

    # Excluindo as naturezas que nao estao mais vigentes
    target.drop(excluir, inplace=True)
    target.reset_index(drop=True, inplace=True)
    data.drop(excluir, inplace=True)
    data.reset_index(drop=True, inplace=True)
    del excluir, sem_relevancia

    return data, target


def data_preparation(data, target, sample=None, test_size=0.3, categorical_columns=None, numerical_columns=None, text_columns=None, tfidf=True):
    # data.columns = [fixColumnName(c) for c in data.columns]

    if sample:
        data = data.sample(sample,)

    data = data.reset_index(drop=True)

    # Seleção dos atributos
    # target = data['valor_saldo_do_empenho']
    # data.drop(['valor_saldo_do_empenho'], axis='columns', inplace=True)
    # numerical_columns.remove('valor_saldo_do_empenho')

    #
    # Criação de meta-atributos e tratamento comum para todos os dados

    # Criação do meta-atributo "Pessoa Jurídica?"
    pessoa_juridica = np.array(
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
    orgao_sucedido = np.zeros(data.shape[0])

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
    acao_programa = np.zeros(data.shape[0], dtype='object')
    for i in range(data.shape[0]):
        acao_programa[i] = (data['acao'].iloc[i] + ' & ' +
                            data['programa'].iloc[i])
    data['acao_programa'] = acao_programa
    data.drop(['acao', 'programa'], axis='columns', inplace=True)
    categorical_columns.remove('acao')
    categorical_columns.remove('programa')
    categorical_columns.append('acao_programa')
    del acao_programa

    # Codigo que mostra a quantidade de empenhos por processo
    quantidade_empenhos_processo = data['empenho_numero_do_processo'].value_counts(
    )
    quantidade_empenhos_processo = quantidade_empenhos_processo.to_dict()
    empenhos_processo = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        empenhos_processo[i] = quantidade_empenhos_processo[data['empenho_numero_do_processo'].iloc[i]]
    data['empenhos_por_processo'] = empenhos_processo
    del empenhos_processo
    del quantidade_empenhos_processo
    data.drop('empenho_numero_do_processo', axis='columns', inplace=True)
    categorical_columns.remove('empenho_numero_do_processo')
    numerical_columns.append('empenhos_por_processo')

    return data, categorical_columns, numerical_columns


def encode_train_test(X_train, X_test, numerical_columns, categorical_columns, text_columns, tfidf=True):
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    X_train, scalers = generate_fit_scaler(X_train, numerical_columns)
    X_test = generate_scaler(X_test, numerical_columns, scalers)

    X_train, ohes = generate_fit_ohe(X_train, categorical_columns)
    X_test = generate_ohe(X_test, categorical_columns, ohes)

    if tfidf:
        X_train, tfvs = generate_fit_tfidf(X_train, text_columns)
        X_test = generate_tfidf(X_test, text_columns, tfvs)

    return X_train, X_test
