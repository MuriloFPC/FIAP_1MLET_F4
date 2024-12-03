# FIAP_1MLET_F4 - Machine Learning Models
Este projeto é parte do curso de Pós-Graduação em MLOps da FIAP. Ele implementa uma pipeline de treinamento de modelos preditivos usando TensorFlow, Keras e outras ferramentas de ciência de dados, com integração ao MLflow para rastreamento de experimentos.

## 📂 Estrutura do Projeto
- **ML_Models/train.py:** Código principal para treinamento do modelo LSTM.
- **app.py:** API para uso em produção, criada com Flask.
- **Utils/utils.py:** Funções utilitárias usadas no projeto.

## Pré-requisitos
- Python 3.10 ou superior

Verifique a versão do Python:
```bash
python --version
```

## Instale as dependências do projeto
Na raiz do projeto, execute:
```bash
pip install -r requirements.txt
```

## Instruções para Inicialização do Ambiente
**1. Configurar o MLflow**
O MLflow é usado para rastrear experimentos e armazenar métricas do treinamento. Certifique-se de que o servidor do MLflow está configurado corretamente:

Inicie o servidor do MLflow:

```bash
mlflow ui
```
Por padrão, ele estará acessível em http://127.0.0.1:5000.

Verifique se a variável de ambiente MLFLOW_TRACKING_URI está configurada para o servidor:
=======
OBS: Devi a limitações do TensorFlow, a versão do Python deve ser 3.11

## Treinamento

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```
Caso esteja no Windows (via CMD):
```bash
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

**2. Rodar o Código de Treinamento**
O treinamento do modelo deve ser iniciado a partir da raiz do projeto para garantir que o módulo Utils.utils seja encontrado.

Execute o código de treinamento:

```bash
python ML_Models/train.py
```

## Consumir o modelo vai API

```bash
python app.py
```
## Endpoints

### /stockLastMonth/{Codigo da Ação}

Retorna o preço de fechamento da ação nos últimos 30 dias.

### /Predict/{Codigo da Ação}

Retorna a previsão do preço de fechamento da ação para os próximos 15 dias.



## Principais Dependências
Python 3.10+
MLflow 2.18.0
TensorFlow 2.18.0
Keras 3.6.0
Flask 3.1.0
yfinance 0.2.50
scikit-learn 1.5.2
pandas 2.2.3
numpy 2.0.2
matplotlib 3.9.2
Para mais detalhes, consulte o arquivo requirements.txt.

## Possíveis Erros e Soluções
**1. Erro ao encontrar o arquivo utils** 
Certifique-se de executar os scripts a partir da raiz do projeto, pois o módulo Utils.utils depende da estrutura de pastas.

**2. Problemas de conexão com o MLflow**
Se encontrar erros como:

```css
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=8080)
```
Certifique-se de que o servidor do MLflow foi iniciado corretamente e que a variável MLFLOW_TRACKING_URI está configurada.\

## 💡 Observações
Sempre rode o código da raiz do projeto para evitar problemas de importação com a pasta Utils.
Certifique-se de que a porta configurada para o MLflow não esteja em uso.

Coleção do Postman
Adicionalmente, existe uma collection do Postman já configurada no repositório, que contém as requisições para interagir com o servidor Flask. Você pode importá-la diretamente para o Postman e usá-la para testar o modelo sem precisar configurar manualmente a requisição.

## 👩‍💻 Autores
=======

## Participantes do Projeto

Nome: Barbara Barreto
Email: barbaraabb19@gmail.com

Nome: Murilo Fischer de Paula Conceição
Email: murilofpc@gmail.com

Nome: Sanderlan Martins da Silva
Email: sanderlanms@gmail.com