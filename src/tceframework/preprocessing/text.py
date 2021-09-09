import re
from unidecode import unidecode
from pandas import DataFrame
from cleantext import clean
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import download
import string


def _clean_partial(text: str) -> str:
    text = clean(text,
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
                 no_punct=False,
                 no_emoji=True,
                 replace_with_url="",
                 replace_with_email="",
                 replace_with_phone_number="",
                 replace_with_number="",
                 replace_with_digit="",
                 replace_with_currency_symbol="",
                 replace_with_punct="",
                 lang="pt",)
    return text


def regularize_columns_name(data: DataFrame) -> DataFrame:

    columns = []
    for column_name in data.columns:
        column_name = column_name.lower().replace(' ', '_')
        column_name = column_name.replace('.', '')
        column_name = unidecode(column_name)
        column_name = re.sub('(\(eof\))|\(|\)', '', column_name)
        column_name = re.sub(
            '(_codigo/nome)|(_cod/nome)|(_dia/mes/ano)|', '', column_name)
        columns.append(column_name)
    data.columns = columns
    return data


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


def remove_punctuation(input_text):
    # Make translation table
    punct = string.punctuation
    # Every punctuation symbol will be replaced by a space
    trantab = str.maketrans(punct, len(punct)*' ')
    return input_text.translate(trantab)


def clean_nlp(text: str) -> str:
    text = remove_punctuation(text)
    return _clean_partial(text)


def clean_tfidf(text: str) -> str:

    text = remove_punctuation(text)
    text = _clean_partial(text,)

    try:
        stopwords_list = stopwords.words('portuguese')
    except:
        download("stopwords")
        download('wordnet')
        stopwords_list = stopwords.words('portuguese')

    stopwords_list.append("pdf")
    stopwords_list.append("total")
    stopwords_list.append("mes")
    stopwords_list.append("goias")
    stopwords_list.append("go")
    text = remove_stopwords(text, stopwords_list)
    text = stemming(text)
    return text
