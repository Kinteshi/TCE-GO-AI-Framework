import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics import f1_score
from tqdm import tqdm
import tcegoframework.config as config
from tcegoframework.io import load_torch_model, printfile, save_torch_model, save_csv_data
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup


class NaturezaClassifier(nn.Module):
    def __init__(self, n_classes: int, pre_trained_model_name: str = config.PRE_TRAINED_MODEL_NAME):
        super(NaturezaClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = self.drop(bert_output['pooler_output'])
        return self.out(output), bert_output['pooler_output']


def train_epoch(model: NaturezaClassifier, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(config.BERT_DEVICE)
        attention_mask = d["attention_mask"].to(config.BERT_DEVICE)
        targets = d["targets"].to(config.BERT_DEVICE)
        outputs, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        predictions.extend(preds)
        real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    macro = f1_score(real_values, predictions, average='macro')
    micro = f1_score(real_values, predictions, average='micro')
    return correct_predictions.double() / n_examples, np.mean(losses), macro, micro


def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    real_values = []
    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(config.BERT_DEVICE)
            attention_mask = d["attention_mask"].to(config.BERT_DEVICE)
            targets = d["targets"].to(config.BERT_DEVICE)
            outputs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            predictions.extend(preds)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    macro = f1_score(real_values, predictions, average='macro')
    micro = f1_score(real_values, predictions, average='micro')
    return correct_predictions.double() / n_examples, np.mean(losses), macro, micro


def fit_bert(model: NaturezaClassifier, epochs: int, train_data_loader: DataLoader, test_data_loader: DataLoader) -> dict:

    # np.random.seed(RANDOM_SEED)
    # torch.manual_seed(RANDOM_SEED)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(config.BERT_DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(config.BERT_DEVICE)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in tqdm(range(epochs)):
        starting = time.time()
        print(f'Epoch {epoch + 1}')
        print('-' * 10)
        train_acc, train_loss, train_macro, train_micro = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            scheduler,
            len(train_data_loader.dataset)
        )
        print(
            f'Train loss {train_loss} macro {train_macro} micro {train_micro}')
        val_acc, val_loss, val_macro, val_micro = eval_model(
            model,
            test_data_loader,
            loss_fn,
            len(test_data_loader.dataset)
        )
        print(
            f'Val   loss {val_loss} macro {val_macro} micro {val_micro}')
        print('',)
        # history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['train_macro'].append(train_macro)
        history['train_micro'].append(train_micro)
        # history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_macro'].append(val_macro)
        history['val_micro'].append(val_micro)

        if val_acc > best_accuracy:
            save_torch_model(model.state_dict(), 'bert_model.bin')
            best_accuracy = val_acc

        print(
            f'Epoch time: {(time.time()-starting)/60}')

    history = DataFrame(history)
    save_csv_data(history, 'bert_history.csv')

    return history


def get_predictions(model: NaturezaClassifier, data_loader: DataLoader,) -> dict:
    model = model.to(config.BERT_DEVICE)
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in tqdm(data_loader):
            texts = d["empenho_text"]
            input_ids = d["input_ids"].to(config.BERT_DEVICE)
            attention_mask = d["attention_mask"].to(config.BERT_DEVICE)
            targets = d["targets"].to(config.BERT_DEVICE)
            outputs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return {'review_texts': review_texts, 'predictions': predictions, 'prediction_probs': prediction_probs, 'real_values': real_values}


# def treinamento_bert(history: dict) -> None:

#     plt.cla()
#     plt.plot(history['train_loss'], label='train loss')
#     plt.plot(history['val_loss'], label='validation loss')
#     plt.title('Training history')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.ylim([0, 1])
#     plt.savefig('Loss.png')

#     plt.cla()
#     plt.plot(history['train_macro'], label='train macro')
#     plt.plot(history['val_macro'], label='validation macro')
#     plt.title('Training history')
#     plt.ylabel('Macro')
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.ylim([0, 1])
#     plt.savefig('Macro.png')

#     plt.cla()
#     plt.plot(history['train_micro'], label='train micro')
#     plt.plot(history['val_micro'], label='validation micro')
#     plt.title('Training history')
#     plt.ylabel('Micro')
#     plt.xlabel('Epoch')
#     plt.legend()
#     plt.ylim([0, 1])
#     plt.savefig('Micro.png')


def generate_bert_representation(model, data_loader) -> DataFrame:
    model = model.to(config.BERT_DEVICE)
    model = model.eval()
    with torch.no_grad():
        outs = []
        for d in tqdm(data_loader):
            text = d['empenho_text']
            input_ids = d['input_ids'].to(config.BERT_DEVICE)
            attention_mask = d['attention_mask'].to(config.BERT_DEVICE)
            _, pooler = model(input_ids, attention_mask)
            outs.extend(pooler)

    representation = pd.DataFrame(
        np.array([tensor.cpu().detach().numpy() for tensor in outs]),
        columns=[f'BERT_{n}' for n in range(0, np.array([tensor.cpu().detach().numpy() for tensor in outs]).shape[1])])

    return representation


def get_saved_model(n_classes: int):
    model = NaturezaClassifier(
        n_classes, config.PRE_TRAINED_MODEL_NAME)
    state_dict = load_torch_model('bert_model.bin')
    model.load_state_dict(state_dict)
    return model
