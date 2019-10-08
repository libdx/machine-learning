import numpy as np
import pandas as pd
import category_encoders as ce

import functools

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

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

@track_time
def load_data():
    pass

@track_time
def _cross_validate():
    pass

@track_time
def evaluate():
    pass

if __name__ == "__main__":
    evaluate()
