from typing import Any

from pandas import DataFrame, Series
from torch import long, tensor
from torch.utils.data import DataLoader, Dataset
import tcegoframework.config as config


class EmpenhoDataset(Dataset):
    def __init__(self, empenho_historico: Series, target: Series, tokenizer):
        self.empenho_historico = empenho_historico
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = config.BERT_MAX_LEN

    def __len__(self) -> int:
        return len(self.empenho_historico)

    def __getitem__(self, item: int) -> dict[str, Any]:
        empenho_historico = str(self.empenho_historico[item])
        target = self.target[item]
        encoding = self.tokenizer.encode_plus(
            empenho_historico,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'empenho_text': empenho_historico,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': tensor(target, dtype=long)
        }


def create_data_loader(data: DataFrame, tokenizer) -> DataLoader:

    dataset = EmpenhoDataset(
        empenho_historico=data.empenho_historico.to_numpy(),
        target=data['natureza_despesa_cod'].to_numpy(),
        tokenizer=tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=config.BERT_BATCH_SIZE,
    )
