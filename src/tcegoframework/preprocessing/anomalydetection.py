from pandas import DataFrame
from tcegoframework.preprocessing.encoders import encode_AD
from tcegoframework.preprocessing.transform import data_preparation


def preprocessing_training_AD(data: DataFrame) -> tuple[DataFrame, DataFrame]:
    cat_col = [
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
        'natureza_despesa_cod',
        'natureza_despesa_nome',
        'formalidade_nome',
        'modalidade_licitacao_nome',
        # 'fonte_recurso_cod',
        'fonte_recurso_nome',
        'beneficiario_cnpj',
        'beneficiario_cpf',
        'beneficiario_cpf/cnpj',
        'periodo',
        'empenho_numero_do_processo',
        # 'empenho_sequencial_empenho.1',
    ]
    text_col = [
        # 'beneficiario_nome',
        'empenho_historico',
    ]
    num_col = [
        'valor_empenhado',
        'valor_anulacao_empenho',
        # 'valor_estorno_anulacao_empenho',
        'valor_cancelamento_empenho',
        # 'valor_anulacao_cancelamento_empenho',
        'valor_saldo_do_empenho',
        'valor_liquidacao_empenho',
        'valor_anulacao_liquidacao_empenho',
        'valor_saldo_liquidado',
        'valor_ordem_de_pagamento',
        'valor_guia_recolhimento',
        'valor_anulacao_ordem_de_pagamento',
        'valor_estorno_anulacao_o_pagamento',
        'valor_estorno_guia_recolhimento',
        'valor_saldo_pago',
        'valor_saldo_a_pagar',
        'valor_a_liquidar',
        'valor_a_pagar_liquidado'
    ]

    data = data.loc[:, (*cat_col,
                        *text_col,
                        *num_col,)]

    X, categorical_columns, numerical_columns = data_preparation(
        data,
        categorical_columns=cat_col,
        numerical_columns=num_col, )

    X = encode_AD(
        X,
        numerical_columns,
        categorical_columns,
        text_col,
        prefix=f'AD',
    )

    return X
