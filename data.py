import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from scipy.stats import moment

from copy import copy

def preprocessing(train, test, scaler = RobustScaler(), RC_concat = True,
                  u_in_cumsum = False,
                  u_in_lag12 = False,
                  u_in_lag34 = False,
                  u_in_lag_back12 = False,
                  u_in_lag_back34 = False,
                  u_in_diff12 = False,
                  u_in_diff34 = False,
                  u_in_diff_back12 = False,
                  u_in_diff_back34 = False,
                  u_in_max = False,
                  u_in_last = False):
    features = ['time_step', 'u_in', 'u_out']
    cat_features = ['R', 'C']
    target = ['pressure']
    
    # Temp time delta features
    """
    df['time_delta'] = df['time_step'].diff()
    df['time_delta'].fillna(0, inplace=True)
    df['time_delta'].mask(df['time_delta'] < 0, 0, inplace=True)

    df['time_delta_1'] = df.groupby('breath_id')['time_delta'].shift(-1)
    df['time_delta_1'].fillna(method='ffill', inplace=True)
    df['time_delta_1'].mask(df['time_delta'] < 0, 0, inplace=True)
    """
    #features += ['time_delta', 'time_delta_1']
    
    if u_in_cumsum:
        for df in [train, test]:
            df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
        features.append('u_in_cumsum')
        
    if u_in_lag12:
        for df in [train, test]:
            df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1).fillna(0)
            df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2).fillna(0)
        features += ['u_in_lag1', 'u_in_lag2']
        
    if u_in_lag34:
        for df in [train, test]:
            df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3).fillna(0)
            df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4).fillna(0)
        features += ['u_in_lag3', 'u_in_lag4']
            
    if u_in_lag_back12:
        for df in [train, test]:
            df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1).fillna(method='ffill')
            df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2).fillna(method='ffill')
        features += ['u_in_lag_back1', 'u_in_lag_back2']
        
    if u_in_lag_back34:
        for df in [train, test]:
            df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3).fillna(method='ffill')
            df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4).fillna(method='ffill')
        features += ['u_in_lag_back3', 'u_in_lag_back4']
        
    if u_in_diff12:
        for df in [train, test]:
            df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
            df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
        features += ['u_in_diff1', 'u_in_diff2']
        
    if u_in_diff34:
        for df in [train, test]:
            df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
            df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
        features += ['u_in_diff3', 'u_in_diff4']
        
    if u_in_diff_back12:
        for df in [train, test]:
            df['u_in_diff_back1'] = df['u_in'] - df['u_in_lag_back1']
            df['u_in_diff_back2'] = df['u_in'] - df['u_in_lag_back2']
        features += ['u_in_diff_back1', 'u_in_diff_back2']
        
    if u_in_diff_back34:
        for df in [train, test]:
            df['u_in_diff_back3'] = df['u_in'] - df['u_in_lag_back3']
            df['u_in_diff_back4'] = df['u_in'] - df['u_in_lag_back4']
        features += ['u_in_diff_back3', 'u_in_diff_back4']
        
    if u_in_max:
        for df in [train, test]:
            df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
            df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
        features += ['breath_id__u_in__max', 'breath_id__u_in__diffmax']
        
    if u_in_last:
        for df in [train, test]:
            # All entries are first point of each breath_id
            first_df = df.loc[0::80,:]
            # All entries are first point of each breath_id
            last_df = df.loc[79::80,:]
            u_in_last_dict = dict(zip(first_df['breath_id'], last_df['u_in']))
            df['u_in_last'] = df['breath_id'].map(u_in_last_dict)
            del u_in_last_dict
        features.append('u_in_last')
        
        
    # Numerical
    """
    if trapezoid:
        for df in [train, test]:
            df['tmp'] = df['time_delta'] * (df['u_in'] + df['u_in_lag1']) / 2
            df['trapezoid'] = df.groupby('breath_id')['tmp'].cumsum()
            df.drop(columns = ['tmp'], inplace = True)  
        features += ['trapezoid']

    if slope:
        for df in [train, test]:
            df['slope'] = (df['u_in_lag_back1'] - df['u_in']) / df['time_delta_1']
        features += ['slope']

    if area_true:
        for df in [train, test]:
            df['tmp'] = df['time_delta'] * df['u_in']
            df['area_true'] = df.groupby('breath_id')['tmp'].cumsum()
            df.drop(columns = ['tmp'], inplace = True)  
        features += ['area_true']
    """

    # Statistical
    """
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['breath_id__u_in__moment2'] = df.groupby(['breath_id'])['u_in'].transform(lambda x: moment(x, moment = 2))
    df['breath_id__u_in__moment3'] = df.groupby(['breath_id'])['u_in'].transform(lambda x: moment(x, moment = 3))
    df['breath_id__u_in__moment4'] = df.groupby(['breath_id'])['u_in'].transform(lambda x: moment(x, moment = 4))
    features += ['breath_id__u_in__max', 'breath_id__u_in__diffmax', 'breath_id__u_in__diffmean', 'breath_id__u_in__moment2', 'breath_id__u_in__moment3', 'breath_id__u_in__moment4']
        """    
    if RC_concat:
        for df in [train, test]:
            df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
            df['R__C'] = df['R__C'].astype(object)
            df.drop(columns = ['R', 'C'], inplace = True)
            # Onehot encode
        train = pd.get_dummies(train, columns = ['R__C'])
        test = pd.get_dummies(test, columns = ['R__C'])
    
    # Scaling
    # Without u_out
    f_u_out = [f for f in features if f != 'u_out']
    train[f_u_out] = scaler.fit_transform(train[f_u_out])
    test[f_u_out] = scaler.transform(test[f_u_out]) 
    
    return train, test, features

def postprocessing(df, k):
    columnnames = [x for x in range(k)]
    # Average by median
    df["pressure"] = df[columnnames].median(axis = 1)
    df.drop(columns = columnnames, inplace = True)
    
    # Round to constants
    train = pd.read_csv("data/train.csv")
    unique_pressures = train["pressure"].unique()
    sorted_pressures = np.sort(unique_pressures)
    total_pressures_len = len(sorted_pressures)
    df["pressure"] = df["pressure"].apply(lambda x: find_nearest(x, sorted_pressures, total_pressures_len))
    
    return df
    
def find_nearest(prediction, sorted_pressures, total_pressures_len):
    # https://www.kaggle.com/snnclsr/a-dummy-approach-to-improve-your-score-postprocess
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        # If the predicted value is bigger than the highest pressure in the train dataset,
        # return the max value.
        return sorted_pressures[-1]
    elif insert_idx == 0:
        # Same control but for the lower bound.
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return lower_val if abs(lower_val - prediction) < abs(upper_val - prediction) else upper_val