import os
import json
from flask_caching import Cache
import numpy as np
from flask import Flask, request, jsonify
import yfinance as yf
from keras.src.saving import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Utils.utils import GetEnvStocks, GetEnvVariable
from datetime import datetime, timedelta
app = Flask(__name__)
config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
app.config.from_mapping(config)
cache = Cache(app)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/stock-last-month/<string:symbol>')
@cache.cached(timeout=300)
def stockLastMonth(symbol):
    if not symbol:
        return jsonify({'error': 'Símbolo da ação é necessário!'}), 400

    try:
        # Calcula a data de início (último mês)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Busca os dados históricos da ação usando yfinance
        stock = yf.Ticker(f'{symbol}.SA')
        history = stock.history(start=start_date, end=end_date)

        # Verifica se há dados retornados
        if history.empty:
            return jsonify({'error': f'Nenhum dado encontrado para {symbol} no último mês.'}), 404

        # Calcula o valor de mercado diário
        market_cap_data = []
        for date, row in history.iterrows():
            close_price = round(row['Close'], 2)  # Arredonda o preço de fechamento
            market_cap_data.append({
                'date': date.strftime('%d/%m/%Y'),
                'close_price': close_price,
            })

        # Resposta formatada
        response = {
            'symbol': symbol,
            'start_date': start_date.strftime('%d/%m/%Y'),
            'end_date': end_date.strftime('%d/%m/%Y'),
            'market_cap_history': market_cap_data
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': 'Erro ao processar os dados.', 'details': str(e)}), 500

@app.route('/predict/<string:stock>')
@cache.cached(timeout=300)
def predict(stock):
    if stock not in modelsDict:
        stocksList = ','.join(modelsDict)
        return f'{stock} não foi treinada, lista de ações treinadas -> {stocksList}', 404

    stock_values = yf.download(f'{stock}.SA', period='6mo')
    stock_values = stock_values['Close'][f'{stock}.SA'].values.reshape(-1, 1)
    stock_values = stock_values[~np.isnan(stock_values)].reshape(-1, 1)
    model = modelsDict[stock]

    inputs = stock_values[len(stock_values) - previous_days - days_to_predict:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.fit_transform(inputs)
    X_test = []
    for i in range(previous_days, previous_days + days_to_predict):
        X_test.append(inputs[i - previous_days:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)

    predicted_stock_price_list = sc.inverse_transform(predicted_stock_price).tolist()
    return jsonify(predicted_stock_price_list), 200

def LoadModel():
    lastTrainingDate = GetEnvVariable('LAST_TRAINING')

    modelsDict = {}
    models_path = 'ML_Models/Models/' + lastTrainingDate
    keras_files = [f for f in os.listdir(models_path) if f.endswith('.keras')]
    print(keras_files)
    for file in keras_files:
        stock = file.replace('.keras', '')
        modelsDict[stock] = load_model(os.path.join(models_path, file))
        print(f'Model for stock {stock} loaded')

    return modelsDict

sc = MinMaxScaler(feature_range=(0, 1))
modelsDict = LoadModel()
previous_days = int(GetEnvVariable('PREVIOUS_DAYS'))
days_to_predict = int(GetEnvVariable('DAYS_TO_PREDICT'))
if __name__ == '__main__':
    app.run()
