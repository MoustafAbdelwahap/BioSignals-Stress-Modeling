from os import times
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from ml_get_X_y_groups_timestamps import *



"""
This file contains different methods to filter the data before using it.
"""



def acc_feature_selection_for_filter() :
    """
    This method was to determine under
    which conditions we could use the accelerometer data to
    filter the hrv data. Filtering the hrv data to keep only the
    "quiet" moments was supposed to make us gain in accuracy.
    Empatica is already filtering the hrv data with the accelerometer, but
    it is hard to say by what criteria.
    """
    # (1) get l2 acc data
    df = get_original_dataframe()["acc"]
    l2_columns = []
    for col in df.columns :
        if str(col).startswith("acc_l2") : 
            l2_columns.append(col)
    # (2) display the values distribution for all columns concerning the l2-norm
    for col in l2_columns :
            plot = sns.displot(df[col])
            plot_height = plot.fig.get_axes()[0].axis()[3]
            average = df[col].mean()
            plt.axvline(x=average, ymin=0, ymax=plot_height, color='red', label="Average")
            quantile = df[col].quantile(q=0.75)
            plt.axvline(x=quantile, ymin=0, ymax=plot_height, color='green', label="3rd quartile")
            plt.legend()
            plt.title(str(col) + " distribution")
            plt.show()
    # (3) compare with hrv values to see from which value in the acc the hrv data is filtered (we use the rmssd as a reference)
    # get acc and hrv data
    df = get_original_dataframe().loc[:, ["acc", "hrv"]]
    df.columns = df.columns.get_level_values(1)
    # for each acc column
    for col in l2_columns :
        # build a dataframe composed of one acc column and the hrv_rmssd column 
        df_composed_of_iteration_acc_column_and_rmssd_column = df.loc[:, [col, "hrv_rmssd"]]
        acc_col_mean = df_composed_of_iteration_acc_column_and_rmssd_column[col].mean()
        # build a sub-df from the previous one, composed by the same columns but with only empty rows for hrv_rmssd 
        nan_df = df_composed_of_iteration_acc_column_and_rmssd_column[df_composed_of_iteration_acc_column_and_rmssd_column['hrv_rmssd'].isna()]
        # prepare plot
        fig, ax = plt.subplots()
        acc_bar = plt.hist(x=df[col], bins=100, label="values distribution") 
        missing_hrv_bar = plt.hist(x=nan_df[col], bins=100, label="where hrv_rmssd was filtered")
        # change remarkable rectangles color to red and note extra-ticks locations for x axis  
        for i in range(len(acc_bar[2])):
            acc_count = acc_bar[2][i].get_height()
            missing_count = missing_hrv_bar[2][i].get_height()
            if acc_count > 0 :
                if missing_count / acc_count > 0.2 :
                    acc_bar[2][i].set_color("red")
        # add extra-ticks on x axis
        # extraticks = []
        # for i in range(len(acc_bar[2])):
        #     if i > 0 and acc_bar[2][i].get_facecolor() != acc_bar[2][i-1].get_facecolor() and len(extraticks) < 4 :
        #         extraticks.append(acc_bar[2][i].get_x())                
        # plt.xticks(rotation=90, fontsize=8)
        # ax.set_xticks(list(ax.get_xticks()) + extraticks)
        # prepare plot
        plt.xlabel(col)
        lim = ax.get_xlim()
        ax.set_xlim(lim)
        plt.ylabel("count")
        # rebuild custom legend
        legend = ax.legend()    
        handles, labels = ax.get_legend_handles_labels()
        handles = [Patch(facecolor='#1f77b4'), handles[1], Patch(facecolor='red')]
        labels.append("at least 20% of the hrv is filtered")
        legend._init_legend_box(handles, labels)
        legend._set_loc(legend._loc)
        legend.set_title(legend.get_title().get_text())
        plt.show()


def acc_l2_mean_distribution() :
    df = get_original_dataframe().loc[:, ["acc", "hrv"]]
    df.columns = df.columns.get_level_values(1)
    col = "acc_l2_mean"
    df_composed_of_iteration_acc_column_and_rmssd_column = df.loc[:, [col, "hrv_rmssd"]]
    acc_col_mean = df_composed_of_iteration_acc_column_and_rmssd_column[col].mean()
    nan_df = df_composed_of_iteration_acc_column_and_rmssd_column[df_composed_of_iteration_acc_column_and_rmssd_column['hrv_rmssd'].isna()]
    fig, ax = plt.subplots()
    acc_bar = plt.hist(x=df[col], bins=100, label="values distribution") 
    missing_hrv_bar = plt.hist(x=nan_df[col], bins=100, label="where hrv_rmssd was filtered")
    for i in range(len(acc_bar[2])):
        acc_count = acc_bar[2][i].get_height()
        missing_count = missing_hrv_bar[2][i].get_height()
        if acc_count > 0 :
            if missing_count / acc_count > 0.2 :
                acc_bar[2][i].set_color("red")
    # add extra-ticks on x axis     
    extraticks = []
    for i in range(len(acc_bar[2])):
        if i > 0 and acc_bar[2][i].get_facecolor() != acc_bar[2][i-1].get_facecolor() and len(extraticks) < 4 :
            extraticks.append(acc_bar[2][i].get_x())                
    plt.xticks(rotation=90, fontsize=8)
    ax.set_xticks(list(ax.get_xticks()) + extraticks)
    plt.xlabel(col)
    lim = ax.get_xlim()
    ax.set_xlim(lim)
    plt.ylabel("count")
    legend = ax.legend()    
    handles, labels = ax.get_legend_handles_labels()
    handles = [Patch(facecolor='#1f77b4'), handles[1], Patch(facecolor='red')]
    labels.append("at least 20% of the hrv is filtered")
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())
    plt.show()



def filter_the_data_according_to_the_accelerometer(X : pd.DataFrame, y : pd.Series, groups : pd.Series, timestamps : pd.Series, acc_column : str, threshold : float, threshold_is_the_max_value : bool) -> pd.DataFrame :
    """
    This method filters the input data according to a column concerning the accelerometer and a threshold value.
    It will eliminate all lines for which the value in the designated column is greater or lower than to the threshold.
    threshold_is_the_max_value = True --> eliminate values above the threshold
    threshold_is_the_max_value = False --> eliminate values under the threshold

    Examples
    --------
    >>> X, y, groups, timestamps = filter_the_data_according_to_the_accelerometer(
            X=X,
            y=y,
            groups=groups,
            timestamps=timestamps,
            acc_column="acc_l2_mean" , # "acc_l2_lineintegral",
            threshold=62, # 0.552 * 10**6 
            threshold_is_the_max_value=False,
    )

    """
    print("\n[Data filtering]")
    # set aside the y distribution
    former_y_dist = y.value_counts()
    # reconstruct index
    X.index = pd.MultiIndex.from_arrays(arrays=[groups, timestamps])
    y.index = pd.MultiIndex.from_arrays(arrays=[groups, timestamps])
    # get acc_column data and keep only the data above the threshold (those to be eliminated)
    acc_series = get_original_dataframe()["acc"][acc_column]
    if threshold_is_the_max_value : 
        acc_series = acc_series[acc_series > threshold]
    else :
        acc_series = acc_series[acc_series < threshold]
    # filter data according to acc column
    temp_index = X.index.intersection(acc_series.index)
    X = X.drop(temp_index)
    y = y.drop(temp_index)
    # print
    print("\t Filter by :", acc_column)
    print("\t Threshold :", threshold)
    print("\t Remaining lines :", len(X.index))
    print("\t Droped lines :", len(timestamps) - len(X.index))
    print("\t Former y distribution :")
    for idx, item in enumerate(former_y_dist) :
        print(f"\t\t {former_y_dist.index[idx]} : {item} values ({round(item / former_y_dist.sum() * 100, 2)} %)")
    new_y_dist = y.value_counts()
    print("\t New y distribution :")
    for idx, item in enumerate(new_y_dist) :
        print(f"\t\t {new_y_dist.index[idx]} : {item} values ({round(item / new_y_dist.sum() * 100, 2)} %)")
    # delete index
    groups = X.index.get_level_values(0).to_series()
    timestamps = X.index.get_level_values(1).to_series()
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    # end
    return X, y, groups, timestamps


def shape_and_sickness_score_distribution() : 
    """
    Displays the distribution of values for the scores of the morning forms 'sickness' and 'shape'.

    """
    df = get_original_dataframe()["survey_answer"]
    lst = [Target_Column.Morning_Sickness, Target_Column.Morning_Shape]
    for x in lst :
        temp = df[x.value]
        temp.dropna(inplace=True)
        sns.displot(temp, kde=True)
        plt.show()


def filter_the_data_according_to_the_sleep_or_sickness_score(X : pd.DataFrame, y : pd.Series, groups : pd.Series, timestamps : pd.Series, acc_column : str, threshold : float) -> pd.DataFrame :
    """

    Examples
    --------
    >>> X, y, groups, timestamps = filter_the_data_according_to_the_sleep_or_sickness_score(
            X=X,
            y=y,
            groups=groups,
            timestamps=timestamps,
            target_column=Target_Column.Morning_Sickness,
            threshold=50,
    )
    """
    print("\n[Data filtering]")
    # TODO



def test() :
    # Get X, y
    X, y, groups, timestamps = get_X_y_groups_timestamps(
        target_column=Target_Column.Noon_Morning_Stress,
        minimum_threshold_for_target_binarization=10,
        maximum_threshold_for_target_binarization=90,
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