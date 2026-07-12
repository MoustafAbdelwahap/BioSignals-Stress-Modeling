import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector



"""
   This file contains methods to make an automatic selection of features for our training. 

    https://scikit-learn.org/stable/modules/feature_selection.html
    https://towardsdatascience.com/5-feature-selection-method-from-scikit-learn-you-should-know-ed4d116e4172
"""


def variance_threshold_feature_selection(X : pd.DataFrame, threshold : float = (.8 * (1 - .8))) -> pd.DataFrame :
    """
    Feature selector that removes all low-variance features (constant in at least 80% of the instances).
    It only sees the input features (X) without considering any information from the dependent variable (y). \n
    Notes : 
        - It is only useful for eliminating features for Unsupervised Modelling rather than Supervised Modelling.
    """
    print("\n[Variance Threshold Feature Selection]")
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    columns_to_keep = X.columns[selector.get_support()]
    to_drop = [x for x in X.columns if x not in set(columns_to_keep)]
    print(f"\t Droped columns ({len(to_drop)}): {to_drop}")
    X = X[columns_to_keep]
    return X



def pairwise_correlation(X : pd.DataFrame, method : str = "pearson") -> pd.DataFrame:
    """
    We want features that have low correlation with one another. If multiple features are highly 
    correlated, then they will not improve our classification performance (and will also slow down training). \n
    """
    print("\n[Pairwise correlation]")
    corr_matrix = X.corr(method=method)
    # drop highly correlated features (greater than 0.95)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # 
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"\t Droped columns ({len(to_drop)}) : {to_drop}")
    # plot correlation matrix
    sns.heatmap(corr_matrix, square=True)
    plt.show()
    X.drop(to_drop, axis=1, inplace=True)
    return X


def features_correlation(dataset : pd.DataFrame, threshold : float) -> pd.DataFrame:
    """
    Find and remove correlated features.
    """
    print("\n[Features correlation]")
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    print(f' \t Correlated features ({len(col_corr)}) : {col_corr}')
    dataset.drop(col_corr, inplace=True, axis=1)
    return dataset


def univariate_feature_selection_SelectKBest(X : pd.DataFrame, y : pd.DataFrame, my_k : int = None, score_function = f_classif) -> pd.DataFrame:
    """
    Works by selecting the best features based on univariate statistical tests.
    These objects take as input a scoring function that returns univariate scores and p-values (or only scores for SelectKBest and SelectPercentile):
    Beware not to use a regression scoring function with a classification problem, you will get useless results.
    Example for classification : chi2, f_classif, mutual_info_classif
    """
    print("\n[Univariate feature selection]")
    selector = SelectKBest(score_function)
    selector.fit(X, y)
    scores = -np.log10(selector.pvalues_)
    scores /= np.max(scores)
    X_indices = np.arange(len(scores))
    plt.bar(X_indices - 0.05, scores, width=0.2)
    plt.title("Univariate feature selection")
    plt.xlabel("Feature number")
    plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=5, title=f"selector=SelectKBest({score_function})\n total number of features {len(X.columns)} ")
    plt.show()
    k = [i for i in range(1, len(X.columns)+1)]
    score_for_each_k = list()
    for i in k : 
        indices = (-scores).argsort()[:i]     # index of the i highest values
        temp_score = 0
        for idx in indices :
            temp_score += scores[idx]
        score_for_each_k.append(temp_score/i)
    plt.plot(k, score_for_each_k) #, marker = 'x')
    plt.title("Univariate feature selection, Optimal k value")
    plt.xlabel("Feature number")
    plt.ylabel("K best univariate score sum / k")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=5, title=f"selector=SelectKBest(f_classif)\n total number of features {len(X.columns)} ")
    plt.show()
    selector = SelectKBest(score_function, k=my_k)
    selector.fit(X, y)
    columns_to_keep = X.columns[selector.get_support()]
    columns_to_drop = [x for x in X.columns if x not in set(columns_to_keep)]
    print("\t Droped columns :",  len(columns_to_drop), ":", columns_to_drop)
    X = X[columns_to_keep]
    return X


def recursive_feature_elimination_RFE(X : pd.DataFrame, y : pd.DataFrame, estimator=SVC(C=1.0, kernel="linear", class_weight='balanced'), step=1, n_features_to_select=None) -> pd.DataFrame:
    """
    RFE is a Feature Selection method utilizing a machine learning model to selecting
    the features by eliminating the least important feature after recursively training. \n
    Notes :
        - rbf impossible : the underlying estimator SVC should have `coef_` or `feature_importances_` attribute
        - rfe.n_features_to_select : default=None, half of the features are selected.
    """
    print("\n[Recursive Feature Elimination (RFE)]")
    rfe_selector = RFE(
        estimator=estimator,
        step=step,
        n_features_to_select=n_features_to_select, # None, half of the features are selected
        verbose=2
    )
    rfe_selector.fit(X, y)
    columns_to_keep = X.columns[rfe_selector.get_support()]
    columns_to_drop = [x for x in X.columns if x not in set(columns_to_keep)]
    print("\t columns_to_drop", len(columns_to_drop), ":", columns_to_drop)
    X = X[columns_to_keep]
    return X


def recursive_feature_elimination_with_cross_validation_RFECV(X : pd.DataFrame, y : pd.DataFrame, estimator=SVC(C=1.0, kernel="linear", class_weight='balanced'), step=1, cv=StratifiedKFold(3), scoring="f1", min_features_to_select=20) -> pd.DataFrame :
    """
    A recursive feature elimination with automatic tuning of the number of features selected with cross-validation.\n
    Notes :
        - rbf impossible : the underlying estimator SVC should have `coef_` or `feature_importances_` attribute
    """
    print("\n[RFE with automatic tuning of the number of features selected with cross-validation]")
    rfecv = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        verbose=2
    )
    rfecv.fit(X, y)
    plt.title(f"REFCV, Optimal number of features : {rfecv.n_features_}")
    plt.xlabel("Number of features selected")
    plt.ylabel(f"Cross validation score")
    plt.plot(
        range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
        rfecv.grid_scores_,
    )
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=5, title=f"estimator={rfecv.estimator}\nscoring={rfecv.scoring}\n{rfecv.cv}")
    plt.show()
    columns_to_keep = X.columns[rfecv.get_support()]
    columns_to_drop = [x for x in X.columns if x not in set(columns_to_keep)]
    print("\t Droped columns :",  len(columns_to_drop), ":", columns_to_drop)
    X = X[columns_to_keep]
    return X


def selectFromModel(X : pd.DataFrame, y : pd.DataFrame, estimator = SVC(C=1.0, kernel="linear", class_weight='balanced'), threshold = None) -> pd.DataFrame :
    """
    Calculates the importance of each feature using an estimator and retains the one with a score above the set threshold.
    SelectFromModel is significantly faster than SFS. Indeed, SelectFromModel only needs to fit a model 
    once, while SFS needs to cross-validate many different models for each of the iterations
    Notes : 
        - SelectFromModel.threshold : default=None, “mean" is used
    """
    print("\n[Feature Selection via SelectFromModel]")
    sfm_selector = SelectFromModel(
        estimator=estimator,
        threshold=threshold,
    ) 
    sfm_selector.fit(X, y)
    computed_threshold = sfm_selector.threshold_
    importance = np.abs(sfm_selector.estimator_.coef_[0])
    feature_names = np.array(X.columns)
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    plt.bar(height=importance, x=range(len(feature_names)))
    plt.axhline(y=computed_threshold, color='r', linestyle='-')
    plt.title(f"SelectFromModel, {np.count_nonzero(sfm_selector.get_support() == True)} features retained")
    plt.ylabel("Feature importances via coefficients")
    my_labels = []
    for i, feature in enumerate(feature_names) :
        if importance[i] >= computed_threshold : 
            my_labels.append(feature)
        else :
            my_labels.append("")
    plt.xticks(ticks=range(len(feature_names)), labels=my_labels, rotation=90, fontsize=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
        fancybox=True, shadow=True, ncol=5, title=f"estimator={estimator} \n total number of features {len(X.columns)} ")
    plt.show()
    columns_to_keep = X.columns[sfm_selector.get_support()]
    columns_to_drop = [x for x in X.columns if x not in set(columns_to_keep)]
    print("\t Droped columns :",  len(columns_to_drop), ":", columns_to_drop)
    X = X[columns_to_keep]
    return X


def sequential_feature_selection(X : pd.DataFrame, y : pd.DataFrame, estimator=SVC(class_weight='balanced'), scoring="f1", n_features_to_select=20, direction="forward", tol : float = 0.1) -> pd.DataFrame :
    """
    SFS is a greedy procedure where, at each iteration, we choose the best new feature to
    add to our selected features based a cross-validation score. That is, we start with 0 features 
    and choose the best single feature with the highest score. The procedure is repeated until we reach the 
    desired number of selected features.
    > We can also go in the reverse direction (backward SFS), i.e. start with all the features
    and greedily choose features to remove one by one
    > In general, forward and backward selection do not yield equivalent results. Also, one may be much faster than
    the other depending on the requested number of selected features: if we have 10 features and ask for 7 selected
    features, forward selection would need to perform 7 iterations while backward selection would only need to perform 3.
    > SFS differs from RFE and SelectFromModel in that it does not require the underlying model to expose 
    a coef_ or feature_importances_ attribute. It may however be slower considering that more models need to be
    evaluated, compared to the other approaches. For example in backward selection, the iteration going from m features
    to m - 1 features using k-fold cross-validation requires fitting m * k models, while RFE would require only a single fit, and
    SelectFromModel always just does a single fit and requires no iterations.
    
    Examples
    --------
    >>> X_train = sequential_feature_selection(
        X=X_train,
        y=y_train,
        estimator=SVC(class_weight='balanced'),
        scoring="f1",
        n_features_to_select=20,
        direction="forward"
    )
    X_test = X_test[X_train.columns.to_list()]
    """
    print("\n[Sequential Feature Selection]")
    sfs = SequentialFeatureSelector(
        estimator=estimator,
        scoring=scoring,
        n_features_to_select=n_features_to_select,
        tol=tol,
        direction=direction,
        n_jobs=-1
    )
    sfs.fit(X, y)
    columns_to_keep = X.columns[sfs.get_support()]
    columns_to_drop = [x for x in X.columns if x not in set(columns_to_keep)]
    print("\t Droped columns :",  len(columns_to_drop), ":", columns_to_drop)
    print("\t Remaining  columns :",  len(columns_to_keep))
    X = X[columns_to_keep]
    return X

