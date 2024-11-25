from flask import Flask, Response, jsonify, request
import io
import nlpnet
import nltk
import pandas as pd
import re
import requests
import sys
from flask_cors import CORS
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from urllib.parse import unquote
from helpers import *

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
BASE_GOV_URL = 'https://dados.gov.br/dados/api/publico/conjuntos-dados'

# Begin of GOV specific functions
def makeGovRequest(url, params):
    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJINWNwUGpicUpFdFdKMjdTSlI5UFNNZUY1N09xQ1lRVWhjejJCa3UyTFNQa01qVVB5VUVTVWhNZFNOZnVhQnZRRmMtZFhYc0ZvUWFDSjVEVCIsImlhdCI6MTcxMDg0Njc4MX0.yzj8qQ2eq28LoKSp2KAOb05MmDSooirEwVB9LCtoDjg"
    response = requests.get(url, params=params, headers={"chave-api-dados-abertos": token})
    return response.json()

def makeShowDatasetUrl(url, id):
    return f"{url}/{id}"

def getCsvSource(sources):
    for source in sources:
        if source['formato'].lower() == 'csv':
            return source
    return None

def getRepos(word):
    repos = []
    for page in range(10):
        params = {
            'isPrivado': 'false',
            'pagina': page + 1,
            'nomeConjuntoDados': word
        }
        response = makeGovRequest(BASE_GOV_URL, params)
        if len(response) == 0:
            break
        repos += response
    return repos
# End of GOV specific functions

# Start of utilities functions
def cleanDataframe(dataframe):
    if not dataframe:
        return dataframe
    dataframe = dataframe.drop_duplicates(subset=["url"])
    dataframe = dataframe.reset_index()
    return dataframe

def getCsv(url):
    df = pd.read_csv(url, encoding='latin1')
    return df
# End of utilities functions

# Start of Hipolita functions
def semanticEnrichmentModule(user_input):
    tagger = nlpnet.POSTagger('pos-pt', language='pt')
    tags = tagger.tag(user_input)
    tags_final = []

    for item in convert(tags):
        for i in item:
            if i[1] == 'N':
                tags_final.append(i[0])
    
    return tags_final

def dataRecoveryModule(enriched_tags):
    empty = True
    final_dataframe, current_dataframe = pd.DataFrame(), pd.DataFrame()
    x, control = 0, 0
    names, urls, format, description = pd.Series(), pd.Series(), pd.Series(), pd.Series()

    for word in enriched_tags:
            
            repos = getRepos(word)

            if len(repos) == 0:
                return repos
            if len(repos) > 0:
                empty = False
                for result in repos:
                    id = result['id']
                    repo = makeGovRequest(makeShowDatasetUrl(BASE_GOV_URL, id), {})
                    resources = repo.get('recursos', False)
                    if resources:
                        source = getCsvSource(repo['recursos'])
                        if source:
                            if len(source['link']) > 0:
                                names.at[x] = repo['titulo']
                                urls.at[x]= source['link']
                                format.at[x] = source['formato']
                                if len(repo['descricao']) > 0:
                                    description.at[x]= repo['descricao']
                                else:
                                    description.at[x]= 'Sem descrição disponível'
                                x = x + 1
                            else:
                                empty = True

            if not empty:
                if len(repos) > 0:
                    current_dataframe = pd.DataFrame({'nome': names, 'url': urls, 'formato': format, 'descricao': description})

                    current_dataframe = current_dataframe.reset_index(drop=True)

                    phrase = ''
                    for index, row in current_dataframe.iterrows():
                        if row["nome"]:
                            if not row["nome"].isalpha():
                                row["nome"] = row["nome"].replace(',', '')
                                row["nome"] = row["nome"].replace('.', '')
                                row["nome"] = row["nome"].replace('-', '')
                                row["nome"] = row["nome"].replace('/', '')
                                row["nome"] = row["nome"].replace('\'', '')
                                phrase = re.split(r'\s', row["nome"].lower())
                        # if row["nome"]:
                        #     if not row["descricao"].isalpha():
                        #         row["descricao"] = row["descricao"].replace(',', '')
                        #         row["descricao"] = row["descricao"].replace('.', '')
                        #         row["descricao"] = row["descricao"].replace('-', '')
                        #         row["descricao"] = row["descricao"].replace('/', '')
                        #         row["descricao"] = row["descricao"].replace('\'', '')
                        #         description = re.split(r'\s', row["descricao"].lower())

                        if word not in phrase:
                            if word not in description:
                                current_dataframe.drop(index)

                        if control == 0:
                            final_dataframe = current_dataframe
                        else:
                            final_dataframe = final_dataframe.append(current_dataframe)

                        control = control + 1
                        x = 0
                    else:
                        control = control + 1
                        x = 0

    return final_dataframe

def dataVisualizationModule(dataframe, dimension, metric):
    fig, axis = plt.subplots()
    axis.plot(dataframe[dimension], dataframe[metric], marker = 'o')
    axis.set_xticks(dataframe[dimension])
    axis.set_yticks(dataframe[metric])
    plt.title(f"{metric} por {dimension}")
    plt.xlabel(dimension)
    plt.ylabel(metric)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return output
# End of Hipolita functions

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"version": "v1.0.0"})

@app.route("/get_datasets", methods=["GET"])
def get_datasets():
    nltk.download('punkt')
    nltk.download('stopwords')
    user_input = request.args.get('user_input')
    enriched_tags = semanticEnrichmentModule(user_input)
    dataframe = dataRecoveryModule(enriched_tags)
    dataframe = cleanDataframe(dataframe)
    if not dataframe:
        return jsonify({"message": "Nenhum dado encontrado"})
    return jsonify(dataframe.to_json(orient="split"))

@app.route("/select_dataset", methods=["GET"])
def select_dataset():
    selected_url = unquote(request.args.get('selected_url'))
    df_csv = getCsv(selected_url)
    return jsonify(df_csv.columns.to_list())

@app.route("/select_columns", methods=["GET"])
def select_columns():
    selected_url = unquote(request.args.get('selected_url'))
    df_csv = getCsv(selected_url)

    dimension = request.args.get('dimension')
    metric = request.args.get('metric')
    arr_dimension = df_csv[dimension].values
    arr_metric = df_csv[metric].values
    df = pd.DataFrame({dimension: arr_dimension, metric: arr_metric}).groupby([dimension]).count().reset_index()
    df = df.rename(columns={metric: 'Quantidade'})
    metric = 'Quantidade'
    output = dataVisualizationModule(df, dimension, metric)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    app.run(port=5000, debug=True)