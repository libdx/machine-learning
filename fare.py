import numpy as np
import pandas as pd
import category_encoders as ce
import geohash2 as geohash
import functools
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

import pdb

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



class NYTaxiFareProcessor:
    GEOHASH_PREFIX = 'dr'
    GEOHASH_PRECISION = 5

    def __init__(self, model, df, scoring):
        self.model = model
        self.df = df
        self.scoring = scoring

    def __parse_date(self, timestamp):
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S UTC")

    def __date_from_row(self, row):
        timestamp = row.pickup_timestamp
        return self.__parse_date(timestamp)

    def __date_series_from_row(self, row):
        d = row.pickup_datetime
        return pd.Series({
            'year': d.year,
            'month': d.month,
            'day': d.day,
            'hour': d.hour + d.minute / 60
        })

    def __parse_geohash(self, lat, lon):
        return geohash.encode(lat, lon, precision=self.GEOHASH_PRECISION)

    def __pickup_geohash_from_row(self, row):
        lat, lon = row.pickup_latitude, row.pickup_longitude
        return self.__parse_geohash(lat, lon)

    def __dropoff_geohash_from_row(self, row):
        lat, lon = row.dropoff_latitude, row.dropoff_longitude
        return self.__parse_geohash(lat, lon)

    @track_time
    def preprocess(self):
        df = self.df

        df.drop(labels='key', axis=1, inplace=True)
        df.rename(columns={'pickup_datetime': 'pickup_timestamp'}, inplace=True)
        df['pickup_datetime'] = df.apply(self.__date_from_row, axis=1)
        df.drop(labels='pickup_timestamp', axis=1, inplace=True)
        df_with_dates = df.apply(self.__date_series_from_row, axis=1)
        df = pd.concat([df, df_with_dates], axis=1)
        df.drop(labels='pickup_datetime', axis=1, inplace=True)
        df['pickup_geohash'] = df.apply(self.__pickup_geohash_from_row, axis=1)
        df['dropoff_geohash'] = df.apply(self.__dropoff_geohash_from_row, axis=1)
        df.drop(labels=[
            'pickup_latitude',
            'pickup_longitude',
            'dropoff_latitude',
            'dropoff_longitude'
        ], axis=1, inplace=True)
        df = df[df.pickup_geohash.str.startswith(self.GEOHASH_PREFIX) &\
           df.dropoff_geohash.str.startswith(self.GEOHASH_PREFIX)]
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
        ord_encoder = ce.OrdinalEncoder(cols=['pickup_geohash', 'dropoff_geohash'])
        df = ord_encoder.fit_transform(df)

        data = df.values
        X = data[:, :-1]
        y = data[:, -1]

        self.X = X
        self.y = y
        #pdb.set_trace()

    @track_time
    def cross_validate(self):
        scores = cross_val_score(self.model, self.X, self.y, cv=7, n_jobs=-1, scoring=self.scoring) 
        return scores

    @track_time
    def run(self):
        self.preprocess()
        return self.cross_validate()

@track_time
def load_data(nrows):
    if nrows is not None:
        df = pd.read_csv('./data/NewYorkCityTaxiFare/train.csv', nrows=nrows)
    else:
        df = pd.read_csv('./data/NewYorkCityTaxiFare/train.csv')
    return df

if __name__ == '__main__':
    model = RandomForestRegressor(n_estimators=10)
    nrows = 5000000
    df = load_data(nrows)
    scoring='neg_mean_squared_error'
    processor = NYTaxiFareProcessor(model=model, df=df, scoring=scoring)
    scores = processor.run()
    rmse = np.sqrt(np.absolute(scores.mean()))
    print(f"""rows number: {nrows},
mean score: {scores.mean()},
score std: {scores.std()},
rmse: {rmse}""")

