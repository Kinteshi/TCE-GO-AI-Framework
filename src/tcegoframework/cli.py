import argparse
import sys

from dateutil.parser import parse

from tcegoframework import config


def parseb(date: str):
    return parse(date, dayfirst=True)


# Define parser
parser = argparse.ArgumentParser(prog='tcegoframework')

# Define args
parser.add_argument(
    'task',
    action='store',
    choices=['training', 'inference'],
    type=str,
    help='''
        Tarefa: obrigatório. Seleciona a tarefa a ser executada.
        \'training\' treina o modelo com todos os dados disponíveis até o dia anterior.
        \'inference\' Realiza inferência nos empenhos do dia anterior por padrão (para mais opções de filtro veja outros parâmetros).
        '''
)

group_date = parser.add_mutually_exclusive_group(required=False)

group_date.add_argument(
    '--daterange', '-dr',
    action='store',
    nargs=2,
    type=parseb,
    help='''
        Filtro por intervalo de datas: opcional. 
        Ao usar essa opção é necessário passar duas datas e todos os empenhos da mais antiga até
        (incluindo) a mais recente serão selecionados. O formato de data preferível é DD/MM/AAAA.
        ''',
)

group_date.add_argument(
    '--dates', '-d',
    action='store',
    nargs='*',
    type=parseb,
    help='''
        Filtro por seleção de datas: opcional.
        Podem ser selecionadas datas individuais para fazer a extração dos empenhos. 
        O formato de datas preferível é DD/MM/AAAA.
        ''',
)

parser.add_argument(
    '--orgaos', '-o',
    action='store',
    type=str,
    nargs='*',
    help='''
        Filtrar por orgão(s): opcional. Só é funcional quando a tarefa selecionada é \'infer\'.
        Podem ser selecionados 1 ou mais órgãos por esse filtro.
        ''',
)


def main():
    args = parser.parse_args()
    filters = {}
    if args.task == 'training':
        if args.daterange:
            filters['daterange'] = args.daterange
        from tcegoframework.flows.training import train_flow
        train_flow(filters)
        sys.exit(0)

    elif args.task == 'inference':
        filters = {}
        if args.dates:
            filters['dates'] = args.dates
        elif args.daterange:
            filters['daterange'] = args.daterange
        if args.orgaos:
            filters['orgaos'] = args.orgaos
        from tcegoframework.flows.inference import inference_flow
        inference_flow(filters)
        sys.exit(0)
