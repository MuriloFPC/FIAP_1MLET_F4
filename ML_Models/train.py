import sys
import os
from datetime import datetime, timedelta

import logging
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error)
import mlflow

from Utils.utils import (GetEnvStocks,
                         GetEnvVariable,
                         SaveEnvVariable)

from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Dropout, Input

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def RegiterMonitor(type, key, value, run_id):
    if (type == 'log'):
        mlflow.log_param(key, value)
    elif (type == 'tag'):
        mlflow.set_tag(key, value)
    elif (type == 'uri'):
        mlflow.set_tracking_uri(uri=value)
    elif (type == 'metric'):
        mlflow.log_metric(key=key, value=value, run_id=run_id)
    elif (type == 'set'):
        return mlflow.set_experiment(value)
    elif (type == 'save'):
        mlflow.keras.log_model(value, key)
    elif (type == 'table'):
        mlflow.log_table(value, key)


def GetRunId():
    run = mlflow.active_run()
    return run.info.run_id


def GetTrainingAndTestSet(stocks):
    logger.info('Getting training and test set')

    stocksAjusted = ' '.join([stock + '.SA' for stock in stocks])

    last_week_date = datetime.now() - timedelta(days=days_to_predict)
    dataset_train = yf.download(stocksAjusted, start='2000-01-01',
                                end=last_week_date.strftime('%Y-%m-%d'))
    dataset_test = yf.download(stocksAjusted, start=last_week_date.strftime(
         '%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    logger.info('Training and test set retrieved')
    return dataset_train, dataset_test


def CreateXYtrain(training_set_scaled):
    X_train = []
    y_train = []
    for i in range(previous_days, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - previous_days:i, 0])
        y_train.append(training_set_scaled[i, 0])
    return np.array(X_train), np.array(y_train)


def CreateRegressor(train_data):
    regressor = Sequential()
    regressor.add(Input(shape=(train_data.shape[1], 1)))

    # Add the LSTM layer and Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor


def SaveModel(regressor, stock):
    regressor.save(f'Models/{today}/{stock}.keras')
    logger.info(f'Model for stock {stock} saved')


def CreateTestPredictions(dataset_train, dataset_test):
    dataset_total = pd.concat(
        [pd.DataFrame(dataset_train), pd.DataFrame(dataset_test)], axis=0)
    inputs = dataset_total[len(dataset_total) -
                           len(dataset_test) -
                           previous_days:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(previous_days, previous_days + len(dataset_test)):
        X_test.append(inputs[i - previous_days:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    return sc.inverse_transform(predicted_stock_price), X_test


def CreatePlot(real_stock_price,
               predicted_stock_price,
               stock,
               dates):
    dates = pd.to_datetime(dates)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, real_stock_price,
             color='red', label=f'{stock} Stock Price')
    plt.plot(dates, predicted_stock_price,
             color='blue', label=f'{stock} Price Prediction')
    plt.title(f'{stock} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{stock} Stock Price')
    plt.legend()

    # Format the dates on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()  # Rotate date labels

    plt.savefig(f'Models/{today}/{stock}.png')


def EvaluateModel(real_stock_price, predicted_stock_price):
    mae = mean_absolute_error(real_stock_price, predicted_stock_price)

    # Calculando RMSE
    rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

    run_id = GetRunId()

    # Calculando MAPE
    mape = mean_absolute_percentage_error(real_stock_price,
                                          predicted_stock_price)
    logger.info(f'MAE: {mae}')
    RegiterMonitor('metric', 'mae', mae, run_id)
    logger.info(f'RMSE: {rmse}')
    RegiterMonitor('metric', 'rmse', rmse, run_id)
    logger.info(f'MAPE: {mape}')
    RegiterMonitor('metric', 'mape', mape, run_id)

    with open(f'Models/{today}/{stock}_evaluation.txt', 'w') as f:
        f.write(f'MAE: {mae}\n')
        f.write(f'RMSE: {rmse}\n')
        f.write(f'MAPE: {mape}\n')


if __name__ == '__main__':
    batch_size = int(GetEnvVariable('BATCH_SIZE'))
    epochs = int(GetEnvVariable('EPOCHS'))
    previous_days = int(GetEnvVariable('PREVIOUS_DAYS'))
    days_to_predict = int(GetEnvVariable('DAYS_TO_PREDICT_TEST'))

    today = datetime.now().strftime('%Y%m%d')
    if not os.path.exists(f'Models/{today}'):
        os.makedirs(f'Models/{today}', exist_ok=True)

    log_dir = '../Logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(log_dir, 'app.log'))

    logger = logging.getLogger('train_model')
    logger.info('Start training model')
    sc = MinMaxScaler(feature_range=(0, 1))
    stocks = GetEnvStocks()
    datasets_train, datasets_test = GetTrainingAndTestSet(stocks)

    try:
        RegiterMonitor('uri', 'uri', 'http://127.0.0.1:8080', 0)
        RegiterMonitor('set', 'set', f'{"MLET1-F4-LSTM"}', 0)
    except mlflow.exceptions.MlflowException as e:
        print(f"Erro ao conectar ao servidor MLflow: {e}")
        logger.error(f"Erro ao conectar ao servidor MLflow: {e}")
        sys.exit(1)
    for stock in stocks:
        logger.info(f'Training model for stock {stock}')

        dataset_train = datasets_train['Close'][f'{stock}.SA'
                                                ].values.reshape(-1, 1)
        dataset_train = dataset_train[~np.isnan(dataset_train)].reshape(-1, 1)

        dataset_test = datasets_test['Close'][f'{stock}.SA'
                                              ].values.reshape(-1, 1)
        dataset_test = dataset_test[~np.isnan(dataset_test)]

        test_dates = datasets_test['Close'][f'{stock}.SA'].keys().to_list()

        training_set_scaled = sc.fit_transform(dataset_train)

        X_train, y_train = CreateXYtrain(training_set_scaled)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        regressor = CreateRegressor(X_train)
        regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        SaveModel(regressor, stock)
        logger.info(f'Model for stock {stock} trained and saved')
        predictPrice, x_test = CreateTestPredictions(dataset_train,
                                                     dataset_test)
        pp = pd.DataFrame(predictPrice, columns=['predict_price'])
        logger.info(f'Predictions for stock {stock} created')

        with mlflow.start_run():
            run_id = GetRunId()
            RegiterMonitor('log', 'epochs', epochs, run_id)
            RegiterMonitor('log', 'previous_days', previous_days, run_id)
            RegiterMonitor('log', 'days_to_predict', days_to_predict, run_id)
            RegiterMonitor('log', 'batch_size', batch_size, run_id)
            runtime = f"{stock}_" + datetime.now().strftime("%Y%m%d_%H%MS")
            RegiterMonitor('log', 'runtime', runtime, run_id)
            RegiterMonitor('tag', 'Training Info', f'{stock}', run_id)
            RegiterMonitor('table', 'predictPrice.json', pp, run_id)
            RegiterMonitor('save', 'regressor', regressor, run_id)

            CreatePlot(dataset_test, predictPrice, stock, test_dates)
            logger.info(f'Plot for stock {stock} created')

            EvaluateModel(dataset_test, predictPrice)
            logger.info(f'Model for stock {stock} evaluated')

            logger.info(f'Training model for stock {stock} finished')
            logger.info('##################################################')

    logger.info('End training model')
    SaveEnvVariable('LAST_TRAINING', today)
