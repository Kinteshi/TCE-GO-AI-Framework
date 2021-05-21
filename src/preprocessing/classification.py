import pandas as pd


def tratarLabel(data):
    label = data['natureza_despesa_cod']
    # pegando as labels e a quantidade de documentos que elas aparecem
    quantidade_labels = pd.DataFrame(label.value_counts(ascending=True))
    # pegando o nome das labels só aparecem uma vez
    label_um_documento_apenas = []
    for i in range(quantidade_labels.shape[0]):
        if(quantidade_labels.iloc[i].values == 1):
            label_um_documento_apenas.append(quantidade_labels.iloc[i].name)
        else:
            break
    # pegando as linhas das classes com só 1 documento
    index_label_unico_documento = []
    for i in range(label.shape[0]):
        if(label.iloc[i] in label_um_documento_apenas):
            index_label_unico_documento.append(i)

    label = label.drop(index_label_unico_documento)
    label.reset_index(drop=True, inplace=True)
    return label, index_label_unico_documento


def label_1_elmento(label):
    quantidade_labels = pd.DataFrame(label.value_counts(ascending=True))
    # pegando o nome das labels só aparecem uma vez
    label_um_documento_apenas = []
    for i in range(quantidade_labels.shape[0]):
        if(quantidade_labels.iloc[i].values == 1):
            label_um_documento_apenas.append(quantidade_labels.iloc[i].name[0])
        else:
            break
    # pegando as linhas das classes com só 1 documento
    index_label_unico_documento = []
    for i in range(label.shape[0]):
        if(label['natureza_despesa_cod'].iloc[i] in label_um_documento_apenas):
            index_label_unico_documento.append(i)

    label = label.drop(index_label_unico_documento)
    label.reset_index(drop=True, inplace=True)
    return label, index_label_unico_documento
