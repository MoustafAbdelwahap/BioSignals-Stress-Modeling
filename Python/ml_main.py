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
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
    LeavePGroupsOut
)
from tqdm import tqdm



from ml_get_X_y_groups_timestamps import *
from ml_split_train_test import *
from ml_data_normalization import *
from ml_model_performance_checking import *
from ml_on_demand_survey import *
from ml_feature_selection import *
from ml_data_filtration import *



from computing_parameters import Computing_Parameters
import my_paths



"""
Tuto pipeline:
    https://scikit-learn.org/stable/modules/compose.html#combining-estimators

Choisir le bon "estimator" :
    https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    Dans notre cas : 
        Ensemble methods :
            averaging methods -> BaggingClassifier, RandomForestClassifier (max_features="sqrt", n_jobs=-1)
            boosting methods -> AdaBoostClassifier, GradientBoostingClassifier
            StackingClassifier (méthode pour combiner des estimateurs afin de réduire leurs biais) -> https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization
        SVC

Benchmark classifieur :
    https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py


Pourquoi un pipeline :
    https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976
    Model selection without nested CV uses the same data to tune model parameters and evaluate model performance. Information may thus “leak" into the model and overfit the data. The magnitude of this effect is primarily dependent on the size of the dataset and the stability of the model.


Exctraction de features avec plusieurs méthode combinées :
    https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py
"""




selected_random_state = 2022






## Test methods  ##

def plot_answers_distribution() :
    df = get_original_dataframe()
    for x in Target_Column :
        if x !=  Target_Column.Morning_and_afternoon_stress_combined :
            sns.displot(df[("survey_answer", x.value)], kde=True)
            plt.show()


def plot_physio_distribution() :
    df = get_original_dataframe()
    df[("date_infos", "rest_day")] = [1 if x else 0 for x in df[("date_infos", "rest_day")]]
    df[("date_infos", "end_of_year_exam")] = [1 if x else 0 for x in df[("date_infos", "end_of_year_exam")]]
    df.columns = df.columns.get_level_values(1) 
    for col in df.column :
        sns.displot(df[col], kde=True)
        plt.show()
        

def RMSSD_distribution() :
    """
    Plot rmssd distribution.
    """
    original = get_original_dataframe()
    original.columns = original.columns.get_level_values(1)

    rmssd = original.loc[:, "hrv_rmssd"]
    rmssd.dropna(inplace=True)
    sns.displot(data=rmssd, kde=True)
    plt.show()

    ln_rmssd = original.loc[:, "hrv_ln_rmssd"]
    ln_rmssd.dropna(inplace=True)
    sns.displot(data=ln_rmssd, kde=True)
    plt.show()


def test_pca() :

    X, y, groups, timestamps = get_X_y_groups_timestamps(
        target_column=Target_Column.Noon_Morning_Stress,
        minimum_threshold_for_target_binarization=20,
        maximum_threshold_for_target_binarization=60
    )
    X = X.values
    
    pca_full = PCA(n_components=None, random_state=selected_random_state)
    pca_full.fit_transform(X)
    print("Variance explained with all features :", sum(pca_full.explained_variance_ratio_ * 100))

    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Explained Variance")
    plt.show()
    for i in [2, 3, 30, 40, 50, 60, 70, 80, 90] :
        print(f"Variance explained with {i} features : {np.cumsum(pca_full.explained_variance_ratio_ * 100)[i-1]}")

    pca_2D = PCA(n_components=2, random_state=selected_random_state)
    X_pca_2 = pca_2D.fit_transform(X)
    sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], hue=y)
    plt.title(f"2D Scattterplot : {round(sum(pca_2D.explained_variance_ratio_ * 100), 2)} % of the variability captured", pad=15)    
    plt.xlabel("pc1")
    plt.ylabel("pc2")

    pca_3D = PCA(n_components=3, random_state=selected_random_state)
    X_pca_3 = pca_3D.fit_transform(X)
    ax = plt.axes(projection="3d")
    ax.scatter3D(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=y)
    plt.title(f"3D Scattterplot : {round(sum(pca_3D.explained_variance_ratio_ * 100), 2)} % of the variability captured", pad=15)    
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.set_zlabel("pc3")


def test_correlation():
    # 1. Import original dataframe
    df = get_original_dataframe()
    df.sort_index(axis=1, inplace=True)
    # 3. Binarize the target (add NaN)
    df[("survey_answer", "Noon_Morning-Stress")] = [0 if x<=20 else (1 if x>=60 else np.NaN) for x in df[("survey_answer", "Noon_Morning-Stress")]]
    # 4. Clean
    df.reset_index(drop=True, inplace=True)
    df.drop(df[df[("survey_answer", "Noon_Morning-Stress")].isna()].index, inplace = True)
    columns_to_drop = []
    for col in df.columns :
        temp = pd.concat([df[col], df[("survey_answer", "Noon_Morning-Stress")]], axis=1)
        incomplete_line_count = len(temp) - len(temp.dropna())
        if incomplete_line_count > 0 :
            if incomplete_line_count == len(df[("survey_answer", "Noon_Morning-Stress")]) :
                columns_to_drop.append(col)
    df = df.drop(columns_to_drop, axis=1)
    df.dropna(inplace=True)
    # 5. Correlation
    X = df.drop([("survey_answer", "Noon_Morning-Stress")], axis=1)
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, square=True)
    plt.show()
    for item in ['acc', 'date_infos', 'eda', 'hrv', 'temp', 'weather'] :
        temp = X[item].corr()
        sns.heatmap(temp, square=True)
        plt.show()


def test_reduce_correlation_with_pca():
    """
    https://medium.com/@andymdc31/using-pca-in-a-machine-learning-pipeline-b6fe3492b1b9

    Be careful, do not fit the PCA on the complete dataframe. At first on 
    "train" then apply the same treatment on "test" :
        pca_95 = PCA(n_components=0.95)
        pca_95.fit(X_train)
        X_train = pd.DataFrame(pca_95.transform(X_train), columns=["PC"+str(x) for x in range(1, len(pca_95.transform(X_train)[0])+1)])
        X_test = pd.DataFrame(pca_95.transform(X_test), columns=["PC"+str(x) for x in range(1, len(pca_95.transform(X_test)[0])+1)])
    """

    df = get_original_dataframe()
    df = df["hrv"]
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)

    scaler = StandardScaler() 
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, square=True)
    plt.show()

    pca_full = PCA(n_components=None, random_state=selected_random_state)
    pca_full.fit(df)
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Explained Variance")
    plt.show()

    pca_95 = PCA(n_components=0.95)
    df = pd.DataFrame(pca_95.fit_transform(df), columns=["PC"+str(x) for x in range(1, len(pca_95.fit_transform(df)[0])+1)])

    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, square=True)
    plt.show()


def classes_distribution_during_time_and_by_user():
    """
    https://scikit-learn.org/stable/modules/cross_validation.html
    This split (StratifiedGroupKFold) is suboptimal in a sense that it might produce imbalanced splits even 
    if perfect stratification is possible. If you have relatively close distribution 
    of classes in each group, using GroupKFold is better.
    """
    # get data
    df = get_original_dataframe()
    df = eliminate_participants_for_whom_we_do_not_have_sufficient_data(original_df=df, minimum_number_of_lines_to_have_per_participant=168)
    df.columns = df.columns.get_level_values(1)
    # reset timezone
    df.index = pd.MultiIndex.from_arrays([df.index.get_level_values(0), df.index.get_level_values(1).tz_localize(None)])
    # retrieve columns of interest
    noon = df["survey_answer_Noon_Morning-Stress"]
    evening = df["survey_answer_Evening_Stress"]
    # binarize the target
    noon = pd.Series([0 if x<=20 else (1 if x>=60 else np.NaN) for x in noon], name=noon.name, index=noon.index)
    noon.dropna(inplace=True)
    evening = pd.Series([0 if x<=20 else (1 if x>=60 else np.NaN) for x in evening], name=evening.name, index=evening.index)
    evening.dropna(inplace=True)
    # reduce data to retrieve only one score per day for each question
    noon.index = pd.MultiIndex.from_arrays([noon.index.get_level_values(0), noon.index.get_level_values(1).floor("D")])
    noon = noon.groupby(by=noon.index.names).first()
    evening.index = pd.MultiIndex.from_arrays([evening.index.get_level_values(0), evening.index.get_level_values(1).floor("D")])
    evening = evening.groupby(by=evening.index.names).first()
    # computes and the proportion of 1 for each day, if we have no value then 0
    experiment_index = pd.date_range(start='2022-02-14', end='2022-05-12')
    noon_temp_list = []
    evening_temp_list = []
    for d in experiment_index :
        if d in noon.index.get_level_values(1) :
            temp = pd.Series(noon.loc[(slice(None) , d)]).value_counts(normalize=True)
            noon_temp_list.append(temp[1.0] if 1.0 in temp.index else 0)
        else : 
            print("no value for", d, "in noon")
            noon_temp_list.append(0)
        if d in evening.index.get_level_values(1) :
            temp = pd.Series(evening.loc[(slice(None) , d)]).value_counts(normalize=True)
            evening_temp_list.append(temp[1.0] if 1.0 in temp.index else 0)
        else : 
            print("no value for", d, "in evening")
            evening_temp_list.append(0)
    noon_fig_during_time, noon_ax_during_time = plt.subplots()
    plt.title("Noon_Morning-Stress during time")
    pd.Series(noon_temp_list, index=experiment_index, name=noon.name).plot()
    noon_ax_during_time.set(
        xlabel="Time",
        ylabel='proportion of the "stressed" class (1)',
        ylim=[0, 1],
    )
    evening_fig_during_time, evening_ax_during_time = plt.subplots()
    plt.title("Evening_Stress during time")
    pd.Series(evening_temp_list, index=experiment_index, name=evening.name).plot(c='c')
    evening_ax_during_time.set(
        xlabel="Time",
        ylabel='proportion of the "stressed" class (1)',
        ylim=[0, 1],
    )
    # compute and plot classe distribution per user
    noon_stressed_proportion_per_user = []
    evening_stressed_proportion_per_user = []
    uuid_min = np.min([np.min(noon.index.get_level_values(0)), np.min(evening.index.get_level_values(0))])
    uuid_max = np.max([np.max(noon.index.get_level_values(0)), np.max(evening.index.get_level_values(0))])
    for uuid in range(uuid_min, uuid_max+1) :
        if uuid in noon.index.get_level_values(0) :
            temp = pd.Series(noon.loc[(uuid, slice(None))]).value_counts(normalize=True)
            noon_stressed_proportion_per_user.append(temp[1.0] if 1.0 in temp.index else 0)
        else :
            print("no value for uuid", uuid, "in noon")
            noon_stressed_proportion_per_user.append(0)
        if uuid in evening.index.get_level_values(0) :
            temp = pd.Series(evening.loc[(uuid, slice(None))]).value_counts(normalize=True)
            evening_stressed_proportion_per_user.append(temp[1.0] if 1.0 in temp.index else 0)
        else : 
            print("no value for uuid", uuid, "in evening")
            evening_stressed_proportion_per_user.append(0)
    noon_fig_per_user, noon_ax_per_user = plt.subplots()
    plt.title("Noon_Morning-Stress per user")
    plt.bar(
        x=np.arange(len(range(uuid_min, uuid_max+1))),
        height=noon_stressed_proportion_per_user
    )
    noon_ax_per_user.set(
        xlabel="Users",
        ylabel='proportion of the "stressed" class (1)',
        ylim=[0, 1],
    )
    evening_fig_per_user, evening_ax_per_user = plt.subplots()
    plt.title("Evening_Stress per user")
    plt.bar(
        x=np.arange(len(range(uuid_min, uuid_max+1))),
        height=evening_stressed_proportion_per_user,
        color = "c"
    )
    evening_ax_per_user.set(
        xlabel="Users",
        ylabel='proportion of the "stressed" class (1)',
        ylim=[0, 1],
    )
    plt.show()


def test_group_cross_validation():
    """
    https://scikit-learn.org/stable/modules/cross_validation.html
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
    """
    # get data
    X, y, groups, timestamps = get_X_y_groups_timestamps(target_column=Target_Column.Noon_Morning_Stress, minimum_threshold_for_target_binarization=20, maximum_threshold_for_target_binarization=60)
    # we round the values of the groups to reduce the number, otherwise it is impossible to display a readable result
    groups = pd.Series([round(x, -1) for x in groups], name=groups.name)
    # parameters
    cmap_data = plt.cm.tab20
    cmap_cv = plt.cm.coolwarm
    n_splits = 4
    line_width = 10
    p_groups_out = 4
    # cross-validation object to test
    cvs = [
        KFold,
        StratifiedKFold,
        GroupKFold,
        StratifiedGroupKFold,
        GroupShuffleSplit,
        LeavePGroupsOut
    ]
    # test loop
    for cv in cvs:
        if cv != LeavePGroupsOut :
            cv = cv(n_splits=n_splits, )
        else :
            cv = cv(n_groups=p_groups_out)
        fig, ax = plt.subplots(figsize=(6, 3))
        # Generate the training/testing visualizations for each CV split
        for idx, (train, test) in enumerate(cv.split(X=X, y=y, groups=groups)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[test] = 1
            indices[train] = 0
            # Visualize the results (folds repartition)
            ax.scatter(
                range(len(indices)),
                [idx + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=line_width,
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )
        # Plot the data classes and groups at the end
        ax.scatter(range(len(X)), [idx + 1.5] * len(X), c=y, marker="_", lw=line_width, cmap=cmap_data)
        ax.scatter(range(len(X)), [idx + 2.5] * len(X), c=groups, marker="_", lw=line_width, cmap=cmap_data)
        # Formatting
        if type(cv) != LeavePGroupsOut :
            yticklabels = list(range(n_splits)) + ["class", "group"]
            ax.set(
                yticks=np.arange(n_splits + 2) + 0.5,
                yticklabels=yticklabels,
                xlabel="Sample index",
                ylabel="CV iteration",
                ylim=[n_splits + 2.2, -0.2],
                xlim=[0, len(X)],
            )
        else : 
            #yticklabels = list(range(n_splits)) + ["class", "group"]
            ax.set(
                #yticks=np.arange(n_splits + 2) + 0.5,
                #yticklabels=yticklabels,
                xlabel="Sample index",
                ylabel="CV iteration",
                #ylim=[n_splits + 2.2, -0.2],
                #xlim=[0, len(X)],
            )
        ax.set_title("{}".format(type(cv).__name__), fontsize=15)
        ax.legend(
            [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
            ["Testing set", "Training set"],
            loc=(1.02, 0.8),
        )
        # Make the legend fit
        plt.tight_layout()
        fig.subplots_adjust(right=0.7)
    plt.show()


def data_presentation(X : pd.DataFrame, y : pd.DataFrame):
    """
    Data presentation
    """
    print(f"\n[Features columns]")
    print(f"\t X shape : {X.shape}")
    print(f"\n[Target column] \n\t {y.name} :")
    target_distribution = y.value_counts()
    for idx, item in enumerate(y.value_counts()) :
        print(f"\t\t {target_distribution.index[idx]} : {item} values ({round(item / len(y) * 100, 2)} %)")
    





## Manage logs ##

def print_both(file, *args):
    """
    Function which prints both to console and to file.
    > Notes :
        - The file argument passed to the function must be opened outside of the function (e.g. at the
        beginning of the program) and closed outside of the function (e.g. at the end of the program).
        You should open it in append mode.
        - Passing *args to the function allows you to pass arguments the same way you do to a print function.
        Example : print_both(open_file_variable, 'pass arguments as if it is', 'print!', 1, '!')
            
    """
    toprint = ' '.join([str(arg) for arg in args])
    print(toprint)
    file.write(toprint)


class Tee(object):
    """
    Context manager that copies stdout and any exceptions to a log file
        https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()




## Hyper-Parameter Tuning and Model Selection ##


"""

    https://towardsdatascience.com/hyper-parameter-tuning-and-model-selection-like-a-movie-star-a884b8ee8d68
    https://scikit-learn.org/stable/modules/grid_search.html
    https://scikit-learn.org/stable/modules/grid_search.html#tips-for-parameter-search
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    https://towardsdatascience.com/shap-for-feature-selection-and-hyperparameter-tuning-a330ec0ea104
    https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
    https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

    A search consists of:
        - an estimator (regressor or classifier such as sklearn.svm.SVC());
        - a parameter space;
        - a method for searching or sampling candidates;
        - a cross-validation scheme
        - a score function.

    Nested CV to calculate real score : 
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py
            # Non_nested parameter search and scoring
            clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=outer_cv)
            clf.fit(X_iris, y_iris)
            non_nested_scores[i] = clf.best_score_

            # Nested CV with parameter optimization
            clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
            nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
            nested_scores[i] = nested_score.mean()

    Possible de train des features selection ET les hyper parametres en meme temps :
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.feature_selection import SelectKBest
    >>> pipe = Pipeline([
    ...    ('select', SelectKBest()),
    ...    ('model', calibrated_forest)])
    >>> param_grid = {
    ...    'select__k': [1, 2],
    ...    'model__base_estimator__max_depth': [2, 4, 6, 8]}
    >>> search = GridSearchCV(pipe, param_grid, cv=5).fit(X, y)

    scoring_metrics = ["balanced_accuracy", "f1", "roc_auc"]
        Lorsque vous spécifiez plusieurs métriques, le refitparamètre doit être défini sur la métrique (chaîne)
        pour laquelle le best_params_sera trouvé et utilisé pour créer le best_estimator_sur l'ensemble de données.
        Si la recherche ne doit pas être ajustée, définissez refit=False. Laisser refit à la valeur par défaut Noneentraînera une erreur
        lors de l'utilisation de plusieurs métriques.


"""


def grid_search_report(cv_results, n_top : int) :
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(cv_results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    cv_results["mean_test_score"][candidate],
                    cv_results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(cv_results["params"][candidate]))
            print("")


def model_selection(use_already_optimized_models : bool) :
    """
    Benchmark example :
        https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

    + use_already_optimized_models : indicate if you want to search by grid to optimize the parameters of 
    the different models or rather use the parameters already optimized by default
    """
    # Logs
    with Tee(os.path.join(my_paths.logs_directory_path, 'model_selection.txt')) :

        # 0. Global parameters
        cross_validation_nbr_splits = 5
        score_metric = "f1"
        nbr_jobs = -1
        verbose_level = 3

        # 1. Get data
        X, y, groups, timestamps = get_X_y_groups_timestamps(
            target_column=Target_Column.Noon_Morning_Stress,
            minimum_threshold_for_target_binarization=20,
            maximum_threshold_for_target_binarization=60
        )
        X = pairwise_correlation(X=X)
        data_presentation(X, y)

        # 2. Split data into separate training and test set
        print("\n[Split data]")
        ratio = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = selected_random_state)
        print(f"\t Ratio : {int(ratio*100)} % \n\t Train : {X_train.shape} \n\t Test {X_test.shape} \n")

        # 3. Search for satisfactory parameters for each model
        grid_search_results = []
        if not use_already_optimized_models :
            print("\n[Search for satisfactory parameters for each model]\n")
            search_space = [
                (
                    "DummyClassifier",
                    DummyClassifier(),
                    {
                        "strategy" : ["most_frequent"],
                        "random_state" : [selected_random_state],
                    }
                ),
                (
                    "RidgeClassifier",
                    RidgeClassifier(),
                    {
                        "alpha" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        "fit_intercept" : [True, False],
                        "tol" : [0.001],
                        "solver" : ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        "class_weight" : ["balanced"],
                        "random_state" : [selected_random_state],
                    }
                ),
                (
                    "RandomForestClassifier",
                    RandomForestClassifier(),
                    {
                        "n_estimators" : list(range(1, 1000, 10)),      
                        "max_depth" : list(range(1, 100, 10)),                         
                        "min_samples_split" : [2, 5, 10],                                                       
                        "min_samples_leaf" : [1, 2, 4],                                                         
                        "bootstrap" :  [True, False],                                                           
                        "class_weight" : ["balanced", "balanced_subsample"],
                        "random_state" : [selected_random_state],
                    }
                ),
                (
                    "KNeighborsClassifier",
                    KNeighborsClassifier(), 
                    {
                        "n_neighbors" : list(range(1,30)),
                        "weights" : ['uniform','distance'],
                        "algorithm" : ['ball_tree', 'kd_tree', 'brute'],
                        "leaf_size" : list(range(1,50)),
                        "metric" : ['minkowski','euclidean','manhattan'],
                        "n_jobs" : [nbr_jobs]
                    }
                ),
                (
                    "RBF_SVC",
                    SVC(),
                    {
                        "kernel" : ["rbf"],
                        "C" : list(range(1, 1000, 10)),
                        "gamma" : [0.001, 0.01, 0.1, 1, 'scale'],
                        "class_weight" : ["balanced"],
                        "random_state" : [selected_random_state],
                    }
                ),
                (
                    "Linear_SVC",
                    SVC(),
                    {
                        "kernel" : ["linear"],
                        "C" : list(range(1, 1000, 10)),
                        "class_weight" : ["balanced"],
                        "random_state" : [selected_random_state],
                    }
                ),
            ]
            for name, estimator, parameters_grid in search_space :
                grid_search = GridSearchCV(
                    estimator=estimator,
                    param_grid=parameters_grid,
                    scoring=score_metric,
                    cv=StratifiedKFold(n_splits=cross_validation_nbr_splits, random_state=selected_random_state, shuffle=True),
                    verbose=verbose_level,
                    n_jobs=nbr_jobs,
                )
                t0_grid_search = time()
                grid_search = grid_search.fit(X_train, y_train)
                t1_grid_search = time()
                grid_search_results.append(
                    {
                        "model_name" : name,
                        "time" : round(t1_grid_search - t0_grid_search, 2),
                        "parameters_grid" : ParameterGrid(param_grid=parameters_grid),
                        "best_estimator" : grid_search.best_estimator_,
                        "best_score" : grid_search.best_score_,
                    }
                )
            for d in grid_search_results :
                print()
                print("=" * 80)
                print(f"{d['model_name']}")
                print(f"\t Gridsearch time : {d['time']} s")
                print(f"\t Best estimator : \n\t\t {d['best_estimator']}")
                print(f"\t Training best score ({score_metric}) : {d['best_score']}")
                print(f"\t Grid search number of test : {cross_validation_nbr_splits * len(d['parameters_grid'])}")
                for key in d['parameters_grid'].param_grid[0] :
                    print("\t\t", key, ":", d['parameters_grid'].param_grid[0][key])
            print("\n[For the next step]\n")
            print("models = [")
            for d in grid_search_results : 
                print(f"\t{(d['model_name'], d['best_estimator'])},")
            print("]")

        # Which models to use for comparison
        models = []
        if use_already_optimized_models : 
            models = [
                    ('DummyClassifier', DummyClassifier(random_state=2022, strategy='most_frequent')),
                    ('RidgeClassifier', RidgeClassifier(alpha=0.5, class_weight='balanced', random_state=2022, solver='lsqr')),
                    ('RandomForestClassifier', RandomForestClassifier(bootstrap=False, class_weight='balanced', max_depth=11, min_samples_leaf=4, n_estimators=541, random_state=2022)),
                    ('KNeighborsClassifier', KNeighborsClassifier(algorithm='ball_tree', leaf_size=1, metric='manhattan', n_jobs=-1, n_neighbors=1)),
                    ('RBF_SVC', SVC(C=11, class_weight='balanced', gamma=0.01, random_state=2022)),
                    ('Linear_SVC', SVC(C=111, class_weight='balanced', kernel='linear', random_state=2022)),
            ]
        else :
             for d in grid_search_results :
                    models.append((d["model_name"], d["best_estimator"]))

        # 4. Model comparison
        comparison_results = []
        for name, m in models : 
            cv_results = cross_validate(
                estimator=m,
                X=X,
                y=y,
                scoring=score_metric,
                cv=StratifiedKFold(n_splits=cross_validation_nbr_splits, random_state=selected_random_state, shuffle=True),
                verbose=verbose_level,
                n_jobs=nbr_jobs, 
            )
            comparison_results.append(
                {
                    "model_name" : name,
                    "fit_time" : round(cv_results["fit_time"].mean(), 2),
                    "score_time" : round(cv_results["score_time"].mean(), 2),
                    "cross_validated_score" : round(cv_results["test_score"].mean(), 2),
                }
            )
        # 5. Presentation
        print("\n[Results presentation]")
        for d in comparison_results :
            print()
            print("=" * 80)
            print(f"{d['model_name']}")
            print(f"\t Fit time : {d['fit_time']}s")
            print(f"\t Score time : {d['score_time']}s")
            print(f"\t Cross validated score : {d['cross_validated_score']}")
        comparison_results.reverse()
        indices = np.arange(len(comparison_results))
        model_name_lst = [x["model_name"] for x in comparison_results]
        training_time_lst = [x["fit_time"] for x in comparison_results]
        training_time_lst /= np.max(training_time_lst)
        test_time_lst = [x["score_time"] for x in comparison_results]
        test_time_lst /= np.max(test_time_lst)
        test_score_lst = [x["cross_validated_score"] for x in comparison_results]
        plt.figure(figsize=(10, 6))
        plt.title("Model selection")
        plt.barh(indices + 0.4, training_time_lst, 0.15, label="fit_time", color="c")
        plt.barh(indices + 0.2, test_time_lst, 0.15, label="score_time", color="navy")
        plt.barh(indices + 0.0, test_score_lst, 0.15, label="cross_validated_score", color="darkorange")
        plt.yticks(())
        plt.legend(loc="best")
        plt.subplots_adjust(left=0.25)
        plt.subplots_adjust(top=0.95)
        plt.subplots_adjust(bottom=0.05)
        for i, c in zip(indices, model_name_lst):
            plt.text(-0.05, i+0.275, c, ha='right', va='center')
        plt.savefig(os.path.join(my_paths.logs_directory_path, 'model_selection.png'))
        plt.show()

        
def svm_optimisation():
    """
    https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
    https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py
    """
    # Logs
    with Tee(os.path.join(my_paths.logs_directory_path, 'svm_optimisation.txt')) :

        # 0. Global parameters
        cross_validation_nbr_splits = 5
        score_metric = "f1"
        nbr_jobs = -1
        verbose_level = 3

        # 1. Get data
        X, y, groups, timestamps = get_X_y_groups_timestamps(
            target_column=Target_Column.Noon_Morning_Stress,
            minimum_threshold_for_target_binarization=20,
            maximum_threshold_for_target_binarization=60
        )
        data_presentation(X, y)

        # 2. Little feature selection
        X = pairwise_correlation(X=X)

        # 3. Define parameters
        param_grid = {
            "kernel" : ["rbf"],
            "C" : np.linspace(start = 1, stop = 1000, num = 100),
            "gamma" : [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
            "class_weight" : [None, "balanced"],
        }

        # 4. Test with StandardScaler
        X_train, X_test, y_train, y_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
        X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
        test1__train_t0 = time()
        grid1 = GridSearchCV(
            estimator=SVC(),
            param_grid=param_grid,
            scoring=score_metric,
            cv=StratifiedGroupKFold(n_splits=cross_validation_nbr_splits, shuffle=True, random_state=selected_random_state),
            verbose=verbose_level,
            n_jobs=nbr_jobs,
        )
        grid1 = grid1.fit(X_train, y_train)
        test1_train_t1 = time()
        test1_test_t0 = time()
        grid1.predict(X_test)
        test1_test_t1 = time()

        # 5. Test with StandardScaler + Personnal data normalization
        X = data_normalization_by_all_nights_average(X=X, groups=groups)
        X_train, X_test, y_train, y_test = get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)
        X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
        test2_t0 = time()
        grid2 = GridSearchCV(
            estimator=SVC(),
            param_grid=param_grid,
            scoring=score_metric,
            cv=StratifiedGroupKFold(n_splits=cross_validation_nbr_splits, shuffle=True, random_state=selected_random_state),
            verbose=verbose_level,
            n_jobs=nbr_jobs,
        )
        grid2 = grid1.fit(X_train, y_train)
        test1 = time()
        test2_test_t0 = time()
        grid2.predict(X_test)
        test2_test_t1 = time()

    

        for clf in [grid1, grid2] :
            print("-"*80)
            print("Best estimator found by grid search : \n", clf.best_estimator_)
            grid_search_report(cv_results=clf.cv_results_, n_top=5)  

        







def test() :

    # Get X, y, groups
    X, y, groups, timestamps = get_X_y_groups_timestamps(
        target_column=Target_Column.Morning_and_afternoon_stress_combined,
        minimum_threshold_for_target_binarization=10,
        maximum_threshold_for_target_binarization=80
    )

    # Data filtration
    # X, y, groups, timestamps = filter_the_data_according_to_the_accelerometer(
    #         X=X,
    #         y=y,
    #         groups=groups,
    #         timestamps=timestamps,
    #         acc_column="acc_l2_lineintegral",
    #         threshold=0.552 * 10**6
    # )

    # Split train/test
    X_train, X_test, y_train, y_test, groups_train, groups_test= get_group_cared_X_train_X_test_y_train_y_test(X=X, y=y, groups=groups)

    # Normalization
    X_train, X_test = classic_data_normalization(X_train=X_train, X_test=X_test)
    # X_train = data_normalization_by_all_nights_average(X=X_train, groups=groups_train)
    # X_test = data_normalization_by_all_nights_average(X=X_test, groups=groups_test)

    # Little hyperparameters optimisation and classifier training
    param_grid = {
            "kernel" : ["rbf"],
            "C" : [1, 10, 100, 100],
            "gamma" : ['auto'],
            "class_weight" : ["balanced"],
            "random_state" : [selected_random_state]
    }
    inner_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=selected_random_state)
    grid = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        scoring="f1",
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

    # Predict Test
    y_pred = grid.best_estimator_.predict(X_test)

    # Check model accurracy
    check_confusion_matrix(y=y_test, y_pred=y_pred)
    check_classification_report(y=y_test, y_pred=y_pred)
    check_roc_auc(estimator=grid.best_estimator_, X_test=X_test, y_test=y_test)


def test_basic():
    selected_random_state = 2022
    selected_target_column=Target_Column.Noon_Morning_Stress
    selected_minimum_threshold_for_target_binarization=0
    selected_maximum_threshold_for_target_binarization=90
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
        scoring="f1",
        cv=inner_CV,
        verbose=3,
        n_jobs=-1,
    )
    fit_time_t1 = time()
    grid.fit(X=X_train, y=y_train, groups=groups_train)
    fit_time_t2 = time()

    check_roc_auc(estimator=grid.best_estimator_, X_test=X_test, y_test=y_test)




if __name__ == '__main__':
    
    # Get X, y
    X, y, groups, timestamps = get_X_y_groups_timestamps(
        target_column=Target_Column.Noon_Morning_Stress,
        minimum_threshold_for_target_binarization=10,
        maximum_threshold_for_target_binarization=90,
    )

    # First little feature selection
    X = variance_threshold_feature_selection(X=X)
    X = pairwise_correlation(X=X)
    # Personnal data normalization
    X = data_normalization_by_all_nights_average(X=X, groups=groups)