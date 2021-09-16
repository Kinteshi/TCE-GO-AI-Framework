# TCE-GO Artificial Intelligence Framework

Framework de Inteligência Artificial desenvolvido para o Tribunal de Contas do Estado de Goiás.

## Instalação

Antes de começar tenha certeza de que o [pip](https://pip.readthedocs.io/en/stable/installing/) está instalado com o [python](https://www.python.org/downloads/)>=3.8.

Para instalar localmente:

    pip install tcegoframework

## Uso

### Requisitos

Um arquivo `config.ini` que possua obrigatoriamente a seguinte estrutura deve existir no diretório onde o framework será chamado:

    [options.dremio]
    connection = database.address:port
    user = username
    password = password

    [options.training]
    expired_class_path = arquivo_naturezas_vigência_expirada.xlsx
    validated_data_path = dados_validados.xlsx

#### `[options.dremio]`

- `connection`: endereço e porta do banco de dados onde serão feitas as consultas.
- `user`: nome de usuário do dremio.
- `password`: senha do usuário do dremio.

#### `[options.training]`

- `expired_class_path`: endereço para o arquivo que possui as naturezas fora de vigência.
- `validated_data_path`: endereço para o arquivo que possui os dados validados por um especialista quanto à corretude.

### Treino

Para iniciar o treinamento:

    tcegoaif training

Será gerada uma estrutura de arquivos que não deve ser modificada para correto funcionamento do framework.

### Inferência

Para que seja possível realizar inferência os arquivos gerados pelo treinamento devem constar no diretório em que o comando for executado:

    tcegoaif inference

Ao final da inferência são gerados dois arquivos:

- `dia-da-execução_results.csv`: resultados da inferência contendo a seguinte estrutura:

    | Identificador do empenho | Empenho Sequencial Empenho | Natureza Predita | Corretude Predita | Avaliação final do modelo |
    |--------------------------|----------------------------|------------------|-------------------|---------------------------|
    | xxxx.xxxx.xxx.xxxxx      | xxxx.xxxx.xxx.xxxxx        | x.x.xx.xx.xx     | OK                | C-M1-M2                   |

- `dia-da-execução_plot.png`: gráfico contendo a distribuição da avaliação final do framework.

### Filtros

Alguns filtros são suportados por ambos os comandos.

- Intervalo de Datas

    Busca todos os documentos no intervalo de datas (dd/mm/aaaa) especificado. Funciona em modo de treino ou inferência e não pode ser usado em conjunto com o filtro de Data.

        tcegoaif inference --daterange 01/01/2020 01/02/2020 

    ou

        tcegoaif inference -dr 01/01/2020 01/02/2020

- Data(s)

    Busca documentos de uma ou mais datas (dd/mm/aaaa) especificadas pelo filtro. Funciona apenas para o modo de inferência e não pode ser usada em conjunto com o filtro de Intervalo de Datas.

        tcegoaif inference --date 01/01/2020 02/01/2020 05/01/2020

    ou

        $tcegoaif inference -d 01/01/2020 02/01/2020 05/01/2020

- Órgão(s)

    Busca documentos de um ou mais órgãos especificados no filtro. Funciona apenas para o modo de inferência e pode ser combinado com qualquer um dos outros filtros.

        tcegoaif inference --orgao FUNEBOM

    ou

        tcegoaif inference -o FUNEBOM

### Contribuições

- Jeferson Marques ([Kinteshi](http://github.com/Kinteshi))
- Thauan Silva ([devthauan](http://github.com/devthauan))
