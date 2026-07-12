import pandas as pd

from sklearn.model_selection import *


"""
This file contains methods to separate a dataset into 
two subsets (train/test) in different ways. 
"""


def train_test_split_description(X_train : pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    """
    Displays a short description of train and test.
    """
    print(f"\t Train ({round(len(y_train)/(len(y_train)+len(y_test))*100, 2)} %) : \n\t\t {len(X_train)} lines \n\t\t 0 : {y_train.value_counts()[0.0]} ({round(y_train.value_counts(normalize=True)[0.0]*100, 2)} %) \n\t\t 1 : {y_train.value_counts()[1.0]} ({round(y_train.value_counts(normalize=True)[1.0]*100, 2)} %)")
    print(f"\t Test ({round(len(y_test)/(len(y_train)+len(y_test))*100, 2)} %) : \n\t\t {len(X_test)} lines \n\t\t 0 : {y_test.value_counts()[0.0]} ({round(y_test.value_counts(normalize=True)[0.0]*100, 2)} %) \n\t\t 1 : {y_test.value_counts()[1.0]} ({round(y_test.value_counts(normalize=True)[1.0]*100, 2)} %)")
        
    
def get_naively_X_train_X_test_y_train_y_test(X : pd.DataFrame, y : pd.DataFrame, ratio : float, stratify : bool, selected_radom_state : int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    """
    Separate X and y in train and test set by considering that all lines are independent.
    The "stratify" parameter allows to balance the repair of classes in train and test.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = get_naively_X_train_X_test_y_train_y_test(
        X=X,
        y=y,
        ratio=0.2,
        stratify=True,
        selected_radom_state=selected_random_state
    )
    """
    print("\n[Naively split data]")
    if stratify : 
        stratify = y
    else :
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, test_size=ratio, random_state=selected_radom_state)
    train_test_split_description(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def get_group_cared_X_train_X_test_y_train_y_test(X : pd.DataFrame, y : pd.DataFrame, groups : pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    """
    Separate X and y in the training and test sets taking into account the uuid of
    the participants as groups. That is, a participant's data will not be able to be found
    in the training AND test sets.

    Attention, si vous voulez vous soucier des groupes et que vous utilisez une recherche
    par grille, n'oubliez pas de le préciser à l'appel de la méthode fit.
    >>> grid.fit(X=X_train, y=y_train, groups=groups_train)

    Examples
    --------
    >>> X_train, X_test, y_train, y_test, groups_train, groups_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
    """
    print("\n[Split data by taking care of the groups]")
    train_inds, test_inds = next(StratifiedGroupKFold().split(X, y,groups=groups))
    X_train, X_test = X.iloc[train_inds], X.iloc[test_inds]
    y_train, y_test = y.iloc[train_inds], y.iloc[test_inds]
    groups_train, groups_test = groups.iloc[train_inds], groups.iloc[test_inds]
    train_test_split_description(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test, groups_train, groups_test 


