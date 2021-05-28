# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from cleantext import clean
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import resample
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

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
data.columns = list(map(fixColumnName, data.columns))
data, _ = filter_tce_data(data, '../database/norel.xlsx')
# data = data.sample(100)

RANDOM_SEED = 15
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
MAX_LEN = 156
BATCH_SIZE = 16
EPOCHS = 10

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clean_text(input_text):
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
    return text


class TCEDataset(Dataset):
    def __init__(self, empenho, targets, tokenizer, max_len):
        self.empenho = empenho
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.empenho)

    def __getitem__(self, item):
        empenho = str(self.empenho[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            empenho,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'empenho_text': empenho,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TCEDataset(
        empenho=df.empenho.to_numpy(),
        targets=df.encodedNatureza.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
    )


class NaturezaClassifier(nn.Module):
    def __init__(self, n_classes):
        super(NaturezaClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = self.drop(bert_output['pooler_output'])
        return self.out(output)

    def get_pooler(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = bert_output['pooler_output']
        return output


df = data[['empenho_historico', 'natureza_despesa_cod']]

df.columns = ['empenho', 'natureza']

df.empenho = df.empenho.apply(clean_text)

lb = LabelEncoder()
lb.classes_ = np.load('../database/labelEncoder.npy', allow_pickle=True)
df['encodedNatureza'] = np.random.randint(1, 650, df.natureza.shape[0])


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

thauan_data_loader = create_data_loader(
    df=df,  # Seu dataframe
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    batch_size=BATCH_SIZE)


model = NaturezaClassifier(len(lb.classes_))
model = model.to(device)

model.load_state_dict(
    torch.load(
        '../database/bert.bin',
        map_location=torch.device(device)
    )
)
model = model.eval()
with torch.no_grad():
    outs = []
    for d in thauan_data_loader:
        text = d['empenho_text']
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        pooler = model.get_pooler(input_ids, attention_mask)
        outs.extend(pooler)

bert_data = pd.DataFrame(
    np.array([tensor.detach().numpy() for tensor in outs]),
    columns=[f'BERT_{n}' for n in range(0, np.array([tensor.detach().numpy() for tensor in outs]).shape[1])])

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
    # 'grupo_despesa',
    # 'elemento_despesa',
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

text_columns = [
    # 'beneficiario_nome',
    # 'empenho_historico',
]

numerical_columns = [
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
    'valor_estorno_anulacao_o._pagamento',
    'valor_estorno_guia_recolhimento',
    'valor_saldo_pago',
    'valor_saldo_a_pagar',
    'valor_a_liquidar',
    'valor_a_pagar_liquidado'
]


target = data.natureza_despesa_cod

data = data.loc[:, (*categorical_columns,
                    *text_columns,
                    *numerical_columns,)]


data = data.reset_index(drop=True).join(bert_data)

data, categorical_columns, numerical_columns = data_preparation(
    data,
    target,
    test_size=0.3,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns,
    text_columns=text_columns,)

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=15, stratify=target)

X_train, X_test = encode_train_test(
    X_train, X_test, numerical_columns, categorical_columns, text_columns, tfidf=False,)

print(f'Data preparation time: {(time.time() - data_prep_time):.2f}s')
print(f'Training shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')
print()

# %%
svm_time = time.time()
print('Training SVR...')

clf = SVC(C=10, kernel='linear', random_state=15)
clf.fit(X_train, y_train)

print(f'SVM training time: {(time.time() - svm_time):.2f}s')
print()
# %%
svm_time = time.time()

y_pred = clf.predict(X_test)

print(f'SVM predict time: {(time.time() - svm_time):.2f}s')
print()
# %%

print(classification_report(y_test, clf.predict(X_test)))
print()

finish_date = datetime.now()
print(f'Finishing evaluation at {finish_date.strftime("%d/%m/%Y %H:%M:%S")}')
