# API - Tech Challenge 

API de consulta dos dados de vinicultura Embrapa.

## Sobre

Este projeto faz parte da quarta entrega do Tech Challenge do curso FIAP MLET - Grupo 46. A API em questão consulta os preços históricos de ações a partir do Yahoo Finance por meio da biblioteca yfinance.

## Framework
O framework escolhido para construçao da API foi o Flask, devido a sua robustez. [Documentação do Flask](https://flask.palletsprojects.com/en/stable/)

## Instalação

Para executar a API, deve-se executar os seguintes passos:

1. Clonar o repositório;
2. Instalar as dependências com `pip install -r requirements.txt`.

## Treinamento

Para treinar o modelo, deve-se executar o seguinte comando:

- `python train.py`

## Uso

Para executar a API, deve-se executar o seguinte comando:

- `python mlflow ui --port 8080`
- `python app`

## Endpoints

### /stockLastMonth/{Codigo da Ação}

Retorna o preço de fechamento da ação nos últimos 30 dias.

### /Predict/{Codigo da Ação}

Retorna a previsão do preço de fechamento da ação para os próximos 15 dias.

## Participantes do Projeto

Nome: Barbara Barreto
Email: barbaraabb19@gmail.com

Nome: Murilo Fischer de Paula Conceição
Email: murilofpc@gmail.com

Nome: Sanderlan Martins da Silva
Email: sanderlanms@gmail.com