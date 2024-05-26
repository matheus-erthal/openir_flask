from flask import Flask, jsonify, request
import requests
import pandas as pd
import re
import nltk
import nlpnet
import time
from helpers import *
import sys

# print("test", file=sys.stderr)

app = Flask(__name__)
BASE_GOV_URL = 'https://dados.gov.br/dados/api/publico/conjuntos-dados'


def makeGovRequest(url, params):
    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJINWNwUGpicUpFdFdKMjdTSlI5UFNNZUY1N09xQ1lRVWhjejJCa3UyTFNQa01qVVB5VUVTVWhNZFNOZnVhQnZRRmMtZFhYc0ZvUWFDSjVEVCIsImlhdCI6MTcxMDg0Njc4MX0.yzj8qQ2eq28LoKSp2KAOb05MmDSooirEwVB9LCtoDjg"
    response = requests.get(url, params=params, headers={"chave-api-dados-abertos": token})
    return response.json()

def showUrl(url, id):
    return f"{url}/{id}"

def getCsvSource(sources):
    for source in sources:
        if source['formato'].lower() == 'csv':
            return source
    return None

def getRepos(palavra):
    repos = []
    for pagina in range(10):
        params = {
            'isPrivado': 'false',
            'pagina': pagina + 1,
            'nomeConjuntoDados': palavra  
        }
        resp = makeGovRequest(BASE_GOV_URL, params)
        if len(resp) == 0:
            break
        repos += resp
    return repos

def getDataframes(tags_final):
    
    vazio = True
    df_final, df = pd.DataFrame(), pd.DataFrame()
    x, y, controle = 0, 0, 0
    nomes, urls, formato, descricao = pd.Series(), pd.Series(), pd.Series(), pd.Series()

    for palavra in tags_final:
            
            repos = getRepos(palavra)

            if len(repos) == 0:
                return repos
            if len(repos) > 0:
                vazio = False
                for resultado in repos:
                    id = resultado['id']
                    repo = makeGovRequest(showUrl(BASE_GOV_URL, id), {})
                    recursos = repo.get('recursos', False)
                    if recursos:
                        source = getCsvSource(repo['recursos'])
                        if source:
                            nomes.at[x] = repo['titulo']
                            urls.at[x]= source['link']
                            formato.at[x] = source['formato']
                            descricao.at[x]= repo['descricao']
                            x = x + 1

            if not vazio:
                # Criando um único dataframe com todos os valores dos 3 atributos
                if len(repos) > 0:
                    df = pd.DataFrame({'nome': nomes, 'url': urls, 'formato': formato, 'descricao': descricao})

                    df = df.reset_index(drop=True)

                    # Refinando o dataframe para conter apenas os dados que realmente possuem o termo desejado (regex)
                    for index, row in df.iterrows():
                        if row["nome"]:
                            if not row["nome"].isalpha():
                                row["nome"] = row["nome"].replace(',', '')
                                row["nome"] = row["nome"].replace('.', '')
                                row["nome"] = row["nome"].replace('-', '')
                                row["nome"] = row["nome"].replace('/', '')
                                row["nome"] = row["nome"].replace('\'', '')
                                frase = re.split(r'\s', row["nome"].lower())
                        if row["nome"]:
                            if not row["descricao"].isalpha():
                                row["descricao"] = row["descricao"].replace(',', '')
                                row["descricao"] = row["descricao"].replace('.', '')
                                row["descricao"] = row["descricao"].replace('-', '')
                                row["descricao"] = row["descricao"].replace('/', '')
                                row["descricao"] = row["descricao"].replace('\'', '')
                                desc = re.split(r'\s', row["descricao"].lower())

                        if palavra not in frase:
                            if palavra not in desc:
                                df.drop(index)

                        if controle == 0:
                            df_final = df
                        else:
                            df_final = df_final.append(df)

                        controle = controle + 1
                        x = 0
                        y = 0
                    else:
                        controle = controle + 1
                        x = 0
                        y = 0

    return df_final

def clean(df_final):

    df_final = df_final.drop_duplicates(keep='first', inplace=False)
    # df_final = df_final.drop_duplicates(subset='url', keep='first', inplace=False)
    # df_final = df_final.drop_duplicates(subset='formato', keep='first', inplace=False)
    # df_final = df_final.drop_duplicates(subset='descricao', keep='first', inplace=False)
    # df_final = df_final.reset_index(drop=True)
    #     lista_termos_nome = linha['nome'].split(" ")
    #     lista_termos_nome = list(map(str.lower, lista_termos_nome))
    #     tags_final = list(map(str.lower, tags_final))
    #     for elem in tags_final:
    #         if elem in lista_termos_nome:
    #             break
    #         else:
    #             lista_termos_desc = linha['descricao'].split(" ")
    #             lista_termos_desc = list(map(str.lower, lista_termos_desc))
    #             for elem in tags_final:
    #                 if elem in lista_termos_desc:
    #                     break
    #                 else:
    #                     df_final = df_final.drop(index)

    return df_final

def setup(user_input):
    if user_input:
        tagger = nlpnet.POSTagger('pos-pt', language='pt')
        tags = tagger.tag(user_input)
        tags_final = []

        for item in convert(tags):
            for i in item:
                if i[1] == 'N':
                    tags_final.append(i[0])

        # # Ignorando os warnings
        # warnings.filterwarnings("ignore")
        # warnings.simplefilter(action='ignore', category=FutureWarning)

        inicio = time.time()

        df_final = getDataframes(tags_final)
        # TODO - tratar os possíveis erros
        df_final = clean(df_final)

        fim = time.time()
        print('duracao: %f segundos' % (fim - inicio), file=sys.stderr)

        df_final.to_csv('results/result.txt')

        df_final = df_final.reset_index()

        return df_final

def selectRow(df, selected_index):
    selected_row = df.loc[selected_index]
    if selected_row.nome:
        return selected_row
    
def getCsv(url):
    df = pd.read_csv(url, encoding='latin1')
    return df

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"version": "v1.0.0"})

@app.route("/get_datasets", methods=["GET"])
def get_datasets():
    nltk.download('punkt')
    nltk.download('stopwords')
    tags_final = request.args.get('tags_final')
    return jsonify(setup(tags_final).to_json(orient="split"))

@app.route("/select_dataset", methods=["GET"])
def select_dataset():
    # TODO - tratar o requisito da seleção ser feita em banco
    selected_index = int(request.args.get('selected_index'))
    tags_final = request.args.get('tags_final')
    nltk.download('punkt')
    nltk.download('stopwords')
    df_final = setup(tags_final)

    selected = selectRow(df_final, selected_index)
    df_csv = getCsv(selected["url"])
    return jsonify(df_csv.columns.to_list())

@app.route("/select_columns", methods=["GET"])
def select_columns():
    # TODO - tratar o requisito da seleção ser feita em banco
    selected_index = int(request.args.get('selected_index'))
    tags_final = request.args.get('tags_final')
    nltk.download('punkt')
    nltk.download('stopwords')
    df_final = setup(tags_final)
    selected = selectRow(df_final, selected_index)
    df_csv = getCsv(selected["url"])

    selected_column1 = request.args.get('selected_column1')
    selected_column2 = request.args.get('selected_column2')
    arr_column1 = df_csv[selected_column1].values
    arr_column2 = df_csv[selected_column2].values
    df_filter = pd.DataFrame({selected_column1: arr_column1, selected_column2: arr_column2})
    return jsonify(df_filter.to_json(orient="split"))

@app.route("/test", methods=["GET"])
def test():
    return "test"

if __name__ == "__main__":
    # Please do not set debug=True in production
    app.run(port=5000, debug=True)