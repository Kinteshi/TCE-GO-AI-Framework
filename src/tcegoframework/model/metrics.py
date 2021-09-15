from pandas import DataFrame
from sklearn.metrics import classification_report
from tcegoframework.io import load_encoder, save_csv_data


def classification_report_csv(y_true, y_pred, encoding: bool, filename: str) -> None:
    if encoding:
        encoder = load_encoder('targetNLP.pkl')
        report = classification_report(
            y_true, y_pred, target_names=encoder.classes_, output_dict=True)
    else:
        report = classification_report(y_true, y_pred, output_dict=True)
    report = DataFrame(report).transpose()
    save_csv_data(report, filename)


def special_report_csv(y_true, y_pred, data: DataFrame, encoding: bool, filename: str) -> None:
    if encoding:
        encoder = load_encoder('targetNLP.pkl')
        report = classification_report(
            y_true, y_pred, target_names=encoder.classes_, output_dict=True)
    else:
        report = classification_report(y_true, y_pred, output_dict=True)
    value_counts = data['natureza_despesa_cod'].value_counts().to_dict()
    for key in report.keys():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            report[key]['n docs'] = value_counts[key]
            report[key]['percent of total'] = value_counts[key] / data.shape[0]
        elif key == 'accuracy':
            continue
        else:
            report[key]['n docs'] = data.shape[0]
            report[key]['percent of total'] = 1
    report = DataFrame(report).transpose()
    save_csv_data(report, filename)
