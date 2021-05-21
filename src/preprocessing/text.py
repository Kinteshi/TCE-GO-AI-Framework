import re


def fixColumnName(input_text):
    input_text = input_text.lower().replace(' ', '_')

    input_text = removeSpecialChars(input_text)

    input_text = re.sub("(\(eof\))|\(|\)", "", input_text)
    input_text = re.sub(
        "(_codigo/nome)|(_cod/nome)|(_dia/mes/ano)|", "", input_text)
    return input_text


def removeSpecialChars(input_text):
    input_text = re.sub(u'[áãâà]', 'a', input_text)
    input_text = re.sub(u'[éèê]', 'e', input_text)
    input_text = re.sub(u'[íì]', 'i', input_text)
    input_text = re.sub(u'[óõôò]', 'o', input_text)
    input_text = re.sub(u'[úùü]', 'u', input_text)
    input_text = re.sub(u'[ç]', 'c', input_text)
    return input_text
