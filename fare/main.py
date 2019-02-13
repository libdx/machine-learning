import os
from math import *
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

# Utils

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

# Geolocation

GEOHASH_PREFIX = 'dr'
GEOHASH_PRECISION = 5

def parse_geohash(lat, lon):
    return geohash.encode(lat, lon, precision=GEOHASH_PRECISION)

def pickup_geohash_from_row(row):
    lat, lon = row.pickup_latitude, row.pickup_longitude
    return parse_geohash(lat, lon)

def dropoff_geohash_from_row(row):
    lat, lon = row.dropoff_latitude, row.dropoff_longitude
    return parse_geohash(lat, lon)

def haversine(coordinate1, coordinate2):
    """Return Haversine distance on Earth in meters for two given coordinates.

    Arguments:
    coordinate1 -- tuple for first point of latitude and longitude given in degrees
    coordinate2 -- tuple for second point of latitude and longitude given in degrees
    """
    lat1, lon1 = coordinate1
    lat2, lon2 = coordinate2

    EARTH_RADIUS = 6371000
    phi1 = radians(lat1)
    phi2 = radians(lat2)

    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0) ** 2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = EARTH_RADIUS * c  
    distance = round(distance, 3)

    return distance


def haversine_from_row(row):
    coordinate1 = row.pickup_latitude, row.pickup_longitude
    coordinate2 = row.dropoff_latitude, row.dropoff_longitude
    return haversine(coordinate1, coordinate2)


@track_time
def clean(df):
    df = df[df.fare_amount >= 0]
    df = df.dropna()
    return df

@track_time
def drop_cols(df):
    df.drop(labels='key', axis=1, inplace=True)
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
        'distance',
        'passenger_count',
        'fare_amount'
    ]]
    return df

@track_time
def transform_geohashes(df):
    df['pickup_geohash'] = df.apply(pickup_geohash_from_row, axis=1)
    df['dropoff_geohash'] = df.apply(dropoff_geohash_from_row, axis=1)
    df = df[df.pickup_geohash.str.startswith(GEOHASH_PREFIX) &\
       df.dropoff_geohash.str.startswith(GEOHASH_PREFIX)]

    ord_encoder = ce.OrdinalEncoder(cols=['pickup_geohash', 'dropoff_geohash'])
    df = ord_encoder.fit_transform(df)
    return df

@track_time
def transform_distances(df):
    df['distance'] = df.apply(haversine_from_row, axis=1)
    return df

@track_time
def transform_dates(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    dt = df['pickup_datetime'].dt

    df['year'] = dt.year
    df['month'] = dt.month
    df['day'] = dt.day
    df['day_of_week'] = dt.dayofweek
    df['hour'] = dt.hour
    return df

@track_time
def transform_geolocation(df):
    df = transform_geohashes(df)
    df = transform_distances(df)
    return df

@track_time
def transform(df):
    df = transform_dates(df)
    df = transform_geolocation(df)
    return df

@track_time
def process(df):
    df = clean(df)
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
    file = './data/train.csv'
    path = os.path.join(os.path.dirname(__file__), file)
    return pd.read_csv(path, nrows=nrows)

@track_time
def eval_random_forest():
    nrows = 10_000

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

def rmse_from(mse):
    return np.sqrt(np.absolute(mse))

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
        print('.', end='')


@track_time
def eval_tensorflow():
    nrows = 50_000
    df = load_data(nrows)
    df = process(df)
    X, y = split(df)
    X = rescale(X)

    epochs = 1_000

    model = build_tensorflow_model([X.shape[1]])
    history = model.fit(
        X, y, epochs=epochs, validation_split = 0.2,
        verbose=0,callbacks=[PrintDot()])

    stats = pd.DataFrame(history.history)
    stats['epoch'] = history.epoch
    stats['val_rmse'] = stats.apply(
        lambda row: rmse_from(row.val_mean_squared_error),
        axis=1
    )
    stats['rmse'] = stats.apply(lambda row: rmse_from(row.mean_squared_error), axis=1)
    print()
    print(stats.tail())

if __name__ == '__main__':
    #eval_random_forest()
    eval_tensorflow()

