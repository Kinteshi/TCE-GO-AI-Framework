import torch

from tcegoframework.cfgparsing import get_random_seed

# Directiores
ROOT_DIR = 'TCEGO_IA_DATA/'
ENC_PATH = ROOT_DIR + 'bin/encoders/'
MODEL_PATH = ROOT_DIR + 'bin/models/'
STDOUT_REDIR_PATH = ROOT_DIR
META_PATH = ROOT_DIR + 'meta/'

# General
RANDOM_SEED = get_random_seed()

# BERT settings
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
BERT_MAX_LEN = 156
BERT_BATCH_SIZE = 16
BERT_LEARNING_RATE = 2e-5
BERT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Columns selection
CLF_CAT = [
    'orgao',
    'orgao_sucessor_atual',
    'tipo_administracao_nome',
    'tipo_poder_nome',
    'funcao',
    'subfuncao',
    'programa',
    'acao',
    'modalidade_licitacao_nome',
    'fonte_recurso_nome',
    'beneficiario_cnpj',
    'beneficiario_cpf',
    'beneficiario_cpf/cnpj',
    'periodo',
    'empenho_numero_do_processo',
]
CLF2_CAT = [
    'orgao',
    'orgao_sucessor_atual',
    'tipo_administracao_nome',
    'tipo_poder_nome',
    'funcao',
    'subfuncao',
    'programa',
    'acao',
    'grupo_despesa',
    'natureza_despesa_cod',
    'modalidade_licitacao_nome',
    'fonte_recurso_nome',
    'beneficiario_cnpj',
    'beneficiario_cpf',
    'beneficiario_cpf/cnpj',
    'periodo',
]
CLF_TEXT = [
    'empenho_historico',
]
CLF2_TEXT = [
    'empenho_historico',
]
CLF_NUM = [
    'valor_empenhado',
    'valor_anulacao_empenho',
    'valor_cancelamento_empenho',
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
CLF2_NUM = [
    'valor_empenhado',
    'valor_anulacao_empenho',
    'valor_cancelamento_empenho',
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


# Inference

# Functions to change constants in runtime

def CHANGE_ROOT_DIR(root_dir: str) -> None:
    global ROOT_DIR, ENC_PATH, MODEL_PATH, STDOUT_REDIR_PATH, META_PATH
    ROOT_DIR = root_dir + '/'
    ENC_PATH = ROOT_DIR + 'bin/encoders/'
    MODEL_PATH = ROOT_DIR + 'bin/models/'
    STDOUT_REDIR_PATH = ROOT_DIR
    META_PATH = ROOT_DIR + 'meta/'
