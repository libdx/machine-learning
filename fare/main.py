import os
import numpy as np
import pandas as pd
import category_encoders as ce
import geohash2 as geohash
import functools
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pdb
#pdb.set_trace()

def track_time(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = f(*args, **kwargs)
        end = datetime.now()
        spent = end - start
        print(f"\"{f.__name__}\" spent time: {spent}")
        return result
    return wrapper



GEOHASH_PREFIX = 'dr'
GEOHASH_PRECISION = 5

def parse_date(timestamp):
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S UTC")

def date_from_row(row):
    timestamp = row.pickup_timestamp
    return parse_date(timestamp)

def date_series_from_row(row):
    d = row.pickup_datetime
    return pd.Series({
        'year': d.year,
        'month': d.month,
        'day': d.day,
        'hour': d.hour + d.minute / 60
    })

def parse_geohash(lat, lon):
    return geohash.encode(lat, lon, precision=GEOHASH_PRECISION)

def pickup_geohash_from_row(row):
    lat, lon = row.pickup_latitude, row.pickup_longitude
    return parse_geohash(lat, lon)

def dropoff_geohash_from_row(row):
    lat, lon = row.dropoff_latitude, row.dropoff_longitude
    return parse_geohash(lat, lon)

@track_time
def rename(df):
    df.rename(columns={'pickup_datetime': 'pickup_timestamp'}, inplace=True)
    return df

@track_time
def drop_cols(df):
    df.drop(labels='key', axis=1, inplace=True)
    df.drop(labels='pickup_timestamp', axis=1, inplace=True)
    df.drop(labels='pickup_datetime', axis=1, inplace=True)
    df.drop(labels=[
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude'
    ], axis=1, inplace=True)
    return df

@track_time
def rescale(X):
    return preprocessing.normalize(X)

@track_time
def select(df):
    df = df[[
        'year',
        'month',
        'day',
        'hour',
        'pickup_geohash',
        'dropoff_geohash',
        'passenger_count',
        'fare_amount'
    ]]
    return df

@track_time
def transform_dates(df):
    df['pickup_datetime'] = df.apply(date_from_row, axis=1)
    df_with_dates = df.apply(date_series_from_row, axis=1)
    df = pd.concat([df, df_with_dates], axis=1)
    return df

@track_time
def transform_geolocation(df):
    df['pickup_geohash'] = df.apply(pickup_geohash_from_row, axis=1)
    df['dropoff_geohash'] = df.apply(dropoff_geohash_from_row, axis=1)
    df = df[df.pickup_geohash.str.startswith(GEOHASH_PREFIX) &\
       df.dropoff_geohash.str.startswith(GEOHASH_PREFIX)]

    ord_encoder = ce.OrdinalEncoder(cols=['pickup_geohash', 'dropoff_geohash'])
    df = ord_encoder.fit_transform(df)
    return df

@track_time
def transform(df):
    df = transform_dates(df)
    df = transform_geolocation(df)
    return df

@track_time
def process(df):
    df = rename(df)
    df = transform(df)
    df = select(df)
    return df

@track_time
def split(df):
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]

    return X, y

@track_time
def cross_validate(model, X, y, scoring):
    scores = cross_val_score(model, X, y, cv=7, n_jobs=-1, scoring=scoring) 
    return scores

@track_time
def load_data(nrows):
    file = './data/NewYorkCityTaxiFare/train.csv'
    path = os.path.join(os.path.dirname(__file__), file)
    return pd.read_csv(path, nrows=nrows)

@track_time
def eval_random_forest():
    nrows = 100000

    df = load_data(nrows)
    df = process(df)
    X, y = split(df)
    X = rescale(X)

    model = RandomForestRegressor(n_estimators=10)
    #model = LinearRegression()
    #model = KNeighborsRegressor()
    scoring='neg_mean_squared_error'

    scores = cross_validate(model, X, y, scoring=scoring)
    rmse = np.sqrt(np.absolute(scores.mean()))
    print(f"""rows number: {nrows}, \nmean score: {scores.mean()}, \nscore std: {scores.std()}, \nrmse: {rmse}""")

@track_time
def build_tensorflow_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=input_shape),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

def rmse(mse):
    return np.sqrt(np.absolute(mse))

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
        print('.', end='')


@track_time
def eval_tensorflow():
    nrows = 5000
    df = load_data(nrows)
    df = process(df)
    X, y = split(df)
    X = rescale(X)

    epochs = 1000

    model = build_tensorflow_model([X.shape[1]])
    history = model.fit(
        X, y, epochs=epochs, validation_split = 0.2,
        verbose=0,callbacks=[PrintDot()])

    stats = pd.DataFrame(history.history)
    stats['epoch'] = history.epoch
    stats['val_rmse'] = stats.apply(lambda row: rmse(row.val_mean_squared_error), axis=1)
    stats['rmse'] = stats.apply(lambda row: rmse(row.mean_squared_error), axis=1)
    print()
    print(stats.tail())

if __name__ == '__main__':
    eval_random_forest()
    #eval_tensorflow()

