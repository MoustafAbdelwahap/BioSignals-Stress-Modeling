import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from ml_get_X_y_groups_timestamps import get_original_dataframe



"""
This file contains different data normalization methods for the MIAMS dataset.
"""


def classic_data_normalization(X_train : pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    """
    Classic data normalization with StandardScaler.
    Notes, y does not need to be normalized since it is already binarized normally.

    Examples
    --------
    >>> X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
    """
    print("\n[Classic data normalization with StandardScaler]")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    print('')
    return X_train, X_test


def data_normalization_by_all_nights_average(X : pd.DataFrame, groups : pd.Series) -> pd.DataFrame :
    """
    Customized data normalization for each participant in the dataframe.
    For each individual, for each feature, calculate the average of the
    values between 2 and 5 am and subtract this value from all others.
    Notes : If column contains infinite values it will be ignored

    Examples
    --------
    >>> X = data_normalization_by_all_nights_average(X=X, groups=groups)
    """
    # load the entire dataframe 
    original_df = get_original_dataframe()
    # restore an index temporarily for X
    X.index = groups
    # for each column
    for col in tqdm(X.columns) :
        # if it concerns a physiological data
        if str(col)[:str(col).index("_")] in ["eda", "hrv", "temp", "acc"] :
            # for each participant
            for uuid in set(groups) :
                # calculation of the average of this features over all the nights
                night_average = original_df.loc[uuid, (slice(None), col)].between_time('3:00', '5:00').mean()[0]
                # finally we subtract the calculated value from all those of this participant
                # X.loc[uuid, col] = [x - night_average for x in X.loc[uuid, col]]
                X.loc[uuid, col] = X.loc[uuid, col] - night_average
    # reset index
    X.reset_index(inplace=True, drop=True)
    # end
    return X


def data_normalization_by_the_last_night_average(X : pd.DataFrame, groups : pd.Series, timestamps : pd.Series) :
    """
    Normalize the physiological data of each user according to the average value
    of the last night. If we don't have values for last night, we take the average 
    over all the user's nights.
    """
    X.index = pd.MultiIndex.from_arrays(arrays=[groups, timestamps])
    original_df = get_original_dataframe()
    col_inf = X.columns.to_series()[np.isinf(X).any()]
    # For each physiological categories
    for col in tqdm(X.columns) :
        if str(col)[:str(col).index("_")] in ["eda", "hrv", "temp", "acc"] :
            # If the column does not contain infinite values
            if col not in col_inf :
                # For each user
                for uuid in set(groups) :
                    # For every day we have data for this user
                    for d in pd.date_range(start=X.loc[uuid, col].index.min().date(), end=X.loc[uuid, col].index.max().date()) :
                        # Calculate last night's average for the column we are processing
                        # If we have no data for last night, we take the average of all the nights
                        last_night_data = original_df.loc[(uuid, str(d.date())), (slice(None), col)].between_time('3:00', '5:00')
                        night_average = None
                        if not last_night_data.empty :
                            night_average = last_night_data.mean()
                        else :     
                            night_average = original_df.loc[uuid, (slice(None), col)].between_time('3:00', '5:00').mean()
                        # Subtract this average from all the values of the day for the column we are processing
                        X.loc[(uuid, str(d.date())), col] = [x - night_average for x in X.loc[(uuid, str(d.date())), col]]
    X.reset_index(drop=True, inplace=True)
    return X

