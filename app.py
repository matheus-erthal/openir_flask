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

def make_gov_request(url, params):
    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJINWNwUGpicUpFdFdKMjdTSlI5UFNNZUY1N09xQ1lRVWhjejJCa3UyTFNQa01qVVB5VUVTVWhNZFNOZnVhQnZRRmMtZFhYc0ZvUWFDSjVEVCIsImlhdCI6MTcxMDg0Njc4MX0.yzj8qQ2eq28LoKSp2KAOb05MmDSooirEwVB9LCtoDjg"
    response = requests.get(url, params=params, headers={"chave-api-dados-abertos": token})
    return response.json()

def show_url(url, id):
    return f"{url}/{id}"

def get_dataframes(tags_final):
    
    vazio = True
    df_final, df = pd.DataFrame(), pd.DataFrame()
    x, y, controle = 0, 0, 0
    nomes, urls, formato, descricao = pd.Series(), pd.Series(), pd.Series(), pd.Series()

    for palavra in tags_final:
            params = {
                'isPrivado': 'false',
                'pagina': 1,
                'nomeConjuntoDados': palavra  
            }

            base_url = 'https://dados.gov.br/dados/api/publico/conjuntos-dados'

            repos = make_gov_request(base_url, params)

            if len(repos) == 0:
                return repos
            if len(repos) > 0:
                vazio = False
                for resultado in repos:
                    id = resultado['id']
                    repo = make_gov_request(show_url(base_url, id), params)
                    nomes.at[x] = repo['titulo']
                    urls.at[x]= repo['recursos'][0]['link']
                    formato.at[x] = repo['recursos'][0]['formato']
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

    print(df_final.to_json, file=sys.stderr)

    return df_final

def clean(tags_final, df_final):

    df_final = df_final.drop_duplicates(subset='nome', keep='first', inplace=False)
    df_final = df_final.drop_duplicates(subset='url', keep='first', inplace=False)
    df_final = df_final.reset_index(drop=True)

    for index, linha in df_final.iterrows():
        lista_termos_nome = linha['nome'].split(" ")
        lista_termos_nome = list(map(str.lower, lista_termos_nome))
        tags_final = list(map(str.lower, tags_final))
        for elem in tags_final:
            if elem in lista_termos_nome:
                break
            else:
                lista_termos_desc = linha['descricao'].split(" ")
                lista_termos_desc = list(map(str.lower, lista_termos_desc))
                for elem in tags_final:
                    if elem in lista_termos_desc:
                        break
                    else:
                        df_final = df_final.drop(index)

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

        df_final = get_dataframes(tags_final)
        # TODO - tratar os possíveis erros
        # df_final = clean(tags_final, df_final)

        fim = time.time()
        print('duracao: %f segundos' % (fim - inicio), file=sys.stderr)

        df_final.to_csv('results/result.txt')

        df_final = df_final.reset_index()

        for index, row in df_final.iterrows():
            print(row["url"], file=sys.stderr)

        return df_final

def selectRow(df, selected_index):
    selected_row = df.loc[selected_index]
    if selected_row.nome.all():
        return selected_row

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
    selected_index = request.args.get('selected_index')
    return jsonify(setup(tags_final).to_json(orient="split"))




if __name__ == "__main__":
    # Please do not set debug=True in production
    app.run(port=5000, debug=True)