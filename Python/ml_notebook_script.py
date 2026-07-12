# External import
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
from time import time
from datetime import datetime, timedelta
import traceback
import sys

from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import *
from tqdm import tqdm

# Personnal import
from ml_get_X_y_groups_timestamps import *
from ml_split_train_test import *
from ml_data_normalization import *
from ml_model_performance_checking import *
from ml_on_demand_survey import *
from ml_feature_selection import *
from ml_data_filtering import *

from computing_parameters import Computing_Parameters
import my_paths

# Plot answers distribution
df = get_original_dataframe()
sns.displot(df[("survey_answer", Target_Column.Noon_Morning_Stress.value)], kde=True)
sns.displot(df[("survey_answer", Target_Column.Evening_Afternoon_Stress.value)], kde=True)
plt.show()

# Global variables
selected_random_state = 2022
selected_target_column=Target_Column.Noon_Morning_Stress # list of posible target values -> for t in Target_Column: print(t)
selected_minimum_threshold_for_target_binarization=10
selected_maximum_threshold_for_target_binarization=90
score_metric = "f1_weighted"

##########################################################################
################################## Exp 1 #################################
##########################################################################
# Classical 80/20 split train/test
# Standard normalization
# clf = SVC(class_weight="balanced")

# Get X, y
X, y, groups, timestamps = get_X_y_groups_timestamps(
    target_column=selected_target_column,
    minimum_threshold_for_target_binarization=selected_minimum_threshold_for_target_binarization,
    maximum_threshold_for_target_binarization=selected_maximum_threshold_for_target_binarization,
)
# Classical 80/20 split train/test
X_train, X_test, y_train, y_test = get_naively_X_train_X_test_y_train_y_test(
    X=X,
    y=y,
    ratio=0.2,
    stratify=True,
    selected_radom_state=selected_random_state
)
# Standard normalization
X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
# Create a classifier (support vector classifier) and train it on the train subset
clf = SVC(class_weight="balanced")
fit_time_t1 = time()
clf.fit(X_train, y_train)
fit_time_t2 = time()
print(f"Fit in {round(fit_time_t2 - fit_time_t1, 2)} s")
# Predict on the test subset
y_pred = clf.predict(X_test)

# Check model accurracy
check_confusion_matrix(y=y_test, y_pred=y_pred)
check_classification_report(y=y_test, y_pred=y_pred, print_bla_bla=False, print_as_text=True)
check_roc_auc(estimator=clf, X_test=X_test, y_test=y_test)


##########################################################################
################################## Exp 2 #################################
##########################################################################
# Groups care 80/20 split train/test (A separation that takes individuals into account and ensures that a participant's data does not end up in the train and test sets)
# Standard normalization
# clf = SVC(class_weight="balanced")

from sklearn.model_selection import StratifiedGroupKFold

# Get X, y
X, y, groups, timestamps = get_X_y_groups_timestamps(
    target_column=selected_target_column,
    minimum_threshold_for_target_binarization=selected_minimum_threshold_for_target_binarization,
    maximum_threshold_for_target_binarization=selected_maximum_threshold_for_target_binarization,
)
# Groups care 80/20 split train/test
X_train, X_test, y_train, y_test, groups_train, groups_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
# Standard normalization
X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
# Create a classifier (support vector classifier) and train it on the train subset
clf = SVC(class_weight="balanced")
fit_time_t1 = time()
clf.fit(X_train, y_train)
fit_time_t2 = time()
print(f"Fit in {round(fit_time_t2 - fit_time_t1, 2)} s")
# Predict on the test subset
y_pred = clf.predict(X_test)

# Check model accurracy
check_confusion_matrix(y=y_test, y_pred=y_pred)
check_classification_report(y=y_test, y_pred=y_pred, print_bla_bla=False, print_as_text=True)
check_roc_auc(estimator=clf, X_test=X_test, y_test=y_test)

##########################################################################
################################## Exp 3 #################################
##########################################################################
# Groups care 80/20 split train/test (A separation that takes individuals into account and ensures that a participant's data does not end up in the train and test sets)
# Standard normalization
# clf = SVC(class_weight="balanced") + Hyperparameter Tuning : scoring_metric="f1_weighted" , inner_CV=StratifiedGroupKFold(n_splits=5)

# Get X, y
X, y, groups, timestamps = get_X_y_groups_timestamps(
    target_column=selected_target_column,
    minimum_threshold_for_target_binarization=selected_minimum_threshold_for_target_binarization,
    maximum_threshold_for_target_binarization=selected_maximum_threshold_for_target_binarization,
)
# Groups care 80/20 split train/test
X_train, X_test, y_train, y_test, groups_train, groups_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
# Standard normalization
X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
# Hyperparameters optimisation and classifier training
param_grid = {
        "kernel" : ["rbf"],
        "C" : [1, 2, 4, 8, 16, 32, 64, 128],
        "gamma" : ['auto', 'scale', 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
        "class_weight" : ["balanced"],
        "random_state" : [selected_random_state]
}
inner_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=selected_random_state)
grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    scoring=score_metric,
    cv=inner_CV,
    verbose=3,
    n_jobs=-1,
)
fit_time_t1 = time()
grid.fit(X=X_train, y=y_train, groups=groups_train)
fit_time_t2 = time()
print("\n[GridSearch results]")
print(f"\t Fit in {round(fit_time_t2 - fit_time_t1, 2)} s")
print(f"\t Train best score : {round(grid.best_score_, 3)} ({grid.scoring})")
print(f"\t Best parameters set for {grid.estimator} :")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t\t %s : %r" % (param_name, best_parameters[param_name]))
# Predict on the test subset
y_pred = grid.best_estimator_.predict(X_test)

# Check model accurracy
check_confusion_matrix(y=y_test, y_pred=y_pred)
check_classification_report(y=y_test, y_pred=y_pred, print_bla_bla=False, print_as_text=True)
check_roc_auc(estimator=grid.best_estimator_, X_test=X_test, y_test=y_test)


##########################################################################
################################## Exp 4 #################################
##########################################################################
# Groups care 80/20 split train/test (A separation that takes individuals into account and ensures that a participant's data does not end up in the train and test sets)
# Standard data normalization
# clf = SVC(class_weight="balanced") + Hyperparameter Tuning : scoring_metric="f1_weighted" , inner_CV=StratifiedGroupKFold(n_splits=5)
# Automatic feature selection :
#     eliminate the features with too little variability
#     those with too much correlation
#     finally we use a sequential selection until the model does not progress anymore.

# Get X, y
X, y, groups, timestamps = get_X_y_groups_timestamps(
    target_column=selected_target_column,
    minimum_threshold_for_target_binarization=selected_minimum_threshold_for_target_binarization,
    maximum_threshold_for_target_binarization=selected_maximum_threshold_for_target_binarization,
)

# First little feature selection
X = variance_threshold_feature_selection(X=X)
X = pairwise_correlation(X=X)
# Groups care 80/20 split train/test
X_train, X_test, y_train, y_test, groups_train, groups_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
# Standard normalization
X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
# Second feature selection
print("\n[Sequential feature selection]")
sfs = SequentialFeatureSelector(
    estimator=SVC(class_weight="balanced"),
    scoring=score_metric,
    n_features_to_select='auto',
    tol=0.001,
    direction="forward",
    cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=selected_random_state),
    n_jobs=-1
)
sfs.fit(X_train, y_train)
columns_to_keep = X.columns[sfs.get_support()]
print(f'\t Direction : {sfs.direction}')
print(f'\t Tol : {sfs.tol}')
print(f'\t Columns to keep ({len(columns_to_keep)}) : {columns_to_keep.to_list()}')
X_train = X_train[columns_to_keep]
X_test = X_test[columns_to_keep]
# Hyperparameters optimisation and classifier training
param_grid = {
        "kernel" : ["rbf"],
        "C" : [1, 2, 4, 8, 16, 32, 64, 128],
        "gamma" : ['auto', 'scale', 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
        "class_weight" : ["balanced"],
        "random_state" : [selected_random_state]
}
inner_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=selected_random_state)
grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    scoring=score_metric,
    cv=inner_CV,
    verbose=3,
    n_jobs=-1,
)
fit_time_t1 = time()
grid.fit(X=X_train, y=y_train, groups=groups_train)
fit_time_t2 = time()
print("\n[GridSearch results]")
print(f"\t Fit in {round(fit_time_t2 - fit_time_t1, 2)} s")
print(f"\t Train best score : {round(grid.best_score_, 3)} ({grid.scoring})")
print(f"\t Best parameters set for {grid.estimator} :")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t\t %s : %r" % (param_name, best_parameters[param_name]))
# Predict on the test subset
y_pred = grid.best_estimator_.predict(X_test)

# Check model accurracy
check_confusion_matrix(y=y_test, y_pred=y_pred)
check_classification_report(y=y_test, y_pred=y_pred, print_bla_bla=False, print_as_text=True)
check_roc_auc(estimator=grid.best_estimator_, X_test=X_test, y_test=y_test)



##########################################################################
################################## Exp 5 #################################
##########################################################################
# Groups care 80/20 split train/test (A separation that takes individuals into account and ensures that a participant's data does not end up in the train and test sets)
# Data normalization according to a baseline established on all the nights of each participant
# clf = SVC(class_weight="balanced") + Hyperparameter Tuning : scoring_metric="f1_weighted" , inner_CV=StratifiedGroupKFold(n_splits=5)
# Little automatic feature selection :
#   eliminate the features with too little variability
#   with too much correlation

# Get X, y
X, y, groups, timestamps = get_X_y_groups_timestamps(
    target_column=selected_target_column,
    minimum_threshold_for_target_binarization=selected_minimum_threshold_for_target_binarization,
    maximum_threshold_for_target_binarization=selected_maximum_threshold_for_target_binarization,
)

# First little feature selection
X = variance_threshold_feature_selection(X=X)
X = pairwise_correlation(X=X)
# Personnal data normalization
X = data_normalization_by_all_nights_average(X=X, groups=groups)
# Groups care 80/20 split train/test
X_train, X_test, y_train, y_test, groups_train, groups_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
# Standard normalization
X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
# Hyperparameters optimisation and classifier training
param_grid = {
        "kernel" : ["rbf"],
        "C" : [1, 2, 4, 8, 16, 32, 64, 128],
        "gamma" : ['auto', 'scale', 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
        "class_weight" : ["balanced"],
        "random_state" : [selected_random_state]
}
inner_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=selected_random_state)
grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    scoring=score_metric,
    cv=inner_CV,
    verbose=3,
    n_jobs=-1,
)
fit_time_t1 = time()
grid.fit(X=X_train, y=y_train, groups=groups_train)
fit_time_t2 = time()
print("\n[GridSearch results]")
print(f"\t Fit in {round(fit_time_t2 - fit_time_t1, 2)} s")
print(f"\t Train best score : {round(grid.best_score_, 3)} ({grid.scoring})")
print(f"\t Best parameters set for {grid.estimator} :")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t\t %s : %r" % (param_name, best_parameters[param_name]))
# Predict on the test subset
y_pred = grid.best_estimator_.predict(X_test)

# Check model accurracy
check_confusion_matrix(y=y_test, y_pred=y_pred)
check_classification_report(y=y_test, y_pred=y_pred, print_bla_bla=False, print_as_text=True)
check_roc_auc(estimator=grid.best_estimator_, X_test=X_test, y_test=y_test)



##########################################################################
################################## Exp 6 #################################
##########################################################################
# Groups care 80/20 split train/test (A separation that takes individuals into account and ensures that a participant's data does not end up in the train and test sets)
# Data filtration according to a threshold
# Data normalization according to a baseline established on all the nights of each participant
# clf = SVC(class_weight="balanced") + Hyperparameter Tuning : scoring_metric="f1_weighted" , inner_CV=StratifiedGroupKFold(n_splits=5)
# Little automatic feature selection :
#   eliminate the features with too little variability
#   with too much correlation

# Get X, y
X, y, groups, timestamps = get_X_y_groups_timestamps(
    target_column=selected_target_column,
    minimum_threshold_for_target_binarization=selected_minimum_threshold_for_target_binarization,
    maximum_threshold_for_target_binarization=selected_maximum_threshold_for_target_binarization,
)
# Data filtration
X, y, groups, timestamps = filter_the_data_according_to_the_accelerometer(
    X=X,
    y=y,
    groups=groups,
    timestamps=timestamps,
    acc_column="acc_l2_lineintegral",
    threshold=0.552 * 10**6, 
    threshold_is_the_max_value=True,
)
# Features selection
X = variance_threshold_feature_selection(X=X)
X = pairwise_correlation(X=X)
# Personnal data normalization
X = data_normalization_by_all_nights_average(X=X, groups=groups)
# Groups care 80/20 split train/test
X_train, X_test, y_train, y_test, groups_train, groups_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
# Standard normalization
X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
# Hyperparameters optimisation and classifier training
param_grid = {
        "kernel" : ["rbf"],
        "C" : [1, 2, 4, 8, 16, 32, 64, 128],
        "gamma" : ['auto', 'scale', 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
        "class_weight" : ["balanced"],
        "random_state" : [selected_random_state]
}
inner_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=selected_random_state)
grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    scoring=score_metric,
    cv=inner_CV,
    verbose=3,
    n_jobs=-1,
)
fit_time_t1 = time()
grid.fit(X=X_train, y=y_train, groups=groups_train)
fit_time_t2 = time()
print("\n[GridSearch results]")
print(f"\t Fit in {round(fit_time_t2 - fit_time_t1, 2)} s")
print(f"\t Train best score : {round(grid.best_score_, 3)} ({grid.scoring})")
print(f"\t Best parameters set for {grid.estimator} :")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t\t %s : %r" % (param_name, best_parameters[param_name]))
# Predict on the test subset
y_pred = grid.best_estimator_.predict(X_test)

# Check model accurracy
check_confusion_matrix(y=y_test, y_pred=y_pred)
check_classification_report(y=y_test, y_pred=y_pred, print_bla_bla=False, print_as_text=True)
check_roc_auc(estimator=grid.best_estimator_, X_test=X_test, y_test=y_test)


##########################################################################
################################## Exp 7 #################################
##########################################################################
# WARNING : For this experiment we use the stress scores from the on-demand forms.
# Groups care 80/20 split train/test (A separation that takes individuals into account and ensures that a participant's data does not end up in the train and test sets)
# Standard data normalization
# clf = SVC(class_weight="balanced") + Hyperparameter Tuning : scoring_metric="f1_weighted" , inner_CV=StratifiedGroupKFold(n_splits=5)
# Little automatic feature selection :
#   eliminate the features with too little variability
#   with too much correlation

# plot distribution
on_demand_survey_distribution()

# Get X, y
X, y, groups, timestamps = get_X_y_groups_timestamps_especially_for_on_demand_survey(
        add_0_artificially=True,
        min_binarization=20,
        max_binarization=60
)
# Little feature selection
X = variance_threshold_feature_selection(X=X)
X = pairwise_correlation(X=X)
# Groups care 80/20 split train/test
X_train, X_test, y_train, y_test, groups_train, groups_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
# Standard normalization
X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
# Hyperparameters optimisation and classifier training
param_grid = {
        "kernel" : ["rbf"],
        "C" : [1, 2, 4, 8, 16, 32, 64, 128],
        "gamma" : ['auto', 'scale', 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
        "class_weight" : ["balanced"],
        "random_state" : [selected_random_state]
}
inner_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=selected_random_state)
grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    scoring=score_metric,
    cv=inner_CV,
    verbose=3,
    n_jobs=-1,
)
fit_time_t1 = time()
grid.fit(X=X_train, y=y_train, groups=groups_train)
fit_time_t2 = time()
print("\n[GridSearch results]")
print(f"\t Fit in {round(fit_time_t2 - fit_time_t1, 2)} s")
print(f"\t Train best score : {round(grid.best_score_, 3)} ({grid.scoring})")
print(f"\t Best parameters set for {grid.estimator} :")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t\t %s : %r" % (param_name, best_parameters[param_name]))
# Predict on the test subset
y_pred = grid.best_estimator_.predict(X_test)

# Check model accurracy
check_confusion_matrix(y=y_test, y_pred=y_pred)
check_classification_report(y=y_test, y_pred=y_pred, print_bla_bla=False, print_as_text=True)
check_roc_auc(estimator=grid.best_estimator_, X_test=X_test, y_test=y_test)
