from enum import Enum
import pandas as pd
import os
import numpy as np


from computing_parameters import Computing_Parameters
import my_paths


"""
This file contains all the methods for reading and preprocessing 
the original dataframe data so that it is ready to use
"""


class Target_Column(Enum):
    """
    List of columns that can be used as labels.
    """
    # Scores completed by participants
    Morning_Shape = "survey_answer_Morning_Shape"
    Morning_Sickness = "survey_answer_Morning_Sickness"
    Morning_Sleep = "survey_answer_Morning_Sleep"
    Noon_Morning_Stress = "survey_answer_Noon_Morning-Stress" 
    Evening_Afternoone_Stress = "survey_answer_Evening_Stress"
    Evening_Negative_Valency = "survey_answer_Evening_Negative_Valency"
    Evening_Positive_Valency = "survey_answer_Evening_Positive_Valency"
    Morning_and_afternoon_stress_combined = "survey_answer_Stress_Score"


def get_original_dataframe() -> pd.DataFrame:
    """
    Returns the "final" dataframe composed of all features hour by hour.
    Ensures that all columns have the appropriate prefix.
    """
    computing_parameters = Computing_Parameters(
        window_length = 3600,
        window_step_size = 3600,
        hrv_threshold = 0.1,
        hrv_clean_data = False,
        only_on_full_hour_slots = True,
    )
    path = os.path.join(my_paths.dataframe_directory_path, my_paths.get_final_parquet_name(computing_parameters=computing_parameters))
    original_df = pd.read_parquet(path)
    level_0 = original_df.columns.get_level_values(0).to_list()
    level_1 = original_df.columns.get_level_values(1).to_list()
    for idx, column_name in enumerate(level_1) :
        if not str(column_name).startswith(str(level_0[idx])+"_") :
            level_1[idx] = str(level_0[idx])+"_"+column_name
    original_df.columns = pd.MultiIndex.from_arrays([level_0, level_1])
    return original_df


def eliminate_participants_for_whom_we_do_not_have_sufficient_data(original_df : pd.DataFrame, minimum_number_of_lines_to_have_per_participant : int) -> pd.DataFrame :
    """
    Eliminate participants for whom we do not have sufficient data.
    """
    print("\n[Eliminate participants with not enought data]")
    uuid_to_drop = []
    for uuid in set(original_df.index.get_level_values(0)) : 
        count = len(original_df.loc[(uuid, slice(None)), :])
        if count < minimum_number_of_lines_to_have_per_participant :
            print(f"\t UUID {uuid}, {count} lines --> removed participant")
            uuid_to_drop.append(uuid)
    original_df.drop(uuid_to_drop, inplace=True)
    return original_df


def clean_index(original_df : pd.DataFrame) -> pd.DataFrame: 
    """
    Changes the multi-index of the columns to a simple index
    and removes the multi-index of the rows.
    """
    print("\n[Index cleaning] \n\t changes the multi-index of the columns to a simple index \n\t removes the multi-index of the rows \n\t\t timestamp and uuid becomes new columns")
    original_df.columns = original_df.columns.get_level_values(1)
    original_df.reset_index(inplace=True)
    return original_df


def build_new_combined_Stress_column(df : pd.DataFrame) -> pd.DataFrame :
    """
    Constructs a new column "Morning_and_afternoon_stress_combined" which
    is the combination of "Noon_Morning_Stress" and "Evening_Afternoone_Stress".
    """
    df[Target_Column.Morning_and_afternoon_stress_combined.value] = np.NaN
    df[Target_Column.Morning_and_afternoon_stress_combined.value].fillna(df[Target_Column.Noon_Morning_Stress.value], inplace=True)
    df[Target_Column.Morning_and_afternoon_stress_combined.value].fillna(df[Target_Column.Evening_Afternoone_Stress.value], inplace=True)
    return df


def drop_unusable_colums(df : pd.DataFrame, target_column : Target_Column) -> pd.DataFrame :
    """
    Drop the unusable columns (not enought values or useless for us).
    """
    print("\n[Drop the unusable columns]")
    original_df = get_original_dataframe()
    acc_columns = original_df.loc[:, "acc"].columns.to_list()
    insufficient_weather_data = ["weather_n", "weather_n_RA"]
    target_incompatible_columns_otherwise_it_s_cheating = original_df.loc[:, "survey_answer"].columns.to_list()
    if target_column != Target_Column.Morning_and_afternoon_stress_combined :
        target_incompatible_columns_otherwise_it_s_cheating.remove(target_column.value)
    to_drop = acc_columns + insufficient_weather_data + target_incompatible_columns_otherwise_it_s_cheating
    df.drop(to_drop, axis=1, inplace=True)
    print(f"\t Dropped columns ({len(to_drop)}) : {to_drop}")   
    return df


def remove_columns_with_infinite_values(df : pd.DataFrame) -> pd.DataFrame:
    """
    Drop dataframe columns if they contain infinite values.
    """
    print("\n[Remove columns with infinite values]")
    df_len = len(df)
    columns_to_drop = []
    for col in df.columns :
        inf_count = np.isinf(df[col]).values.sum()
        if inf_count > 0 :
            print(f"\t {inf_count} infinite values found in {col} ({round(inf_count / df_len * 100, 2)} %) -> column droped")
            columns_to_drop.append(col)
    df = df.drop(columns_to_drop, axis=1)
    return df


def drop_lines_without_value_for_target_column(df : pd.DataFrame, target_column : Target_Column) -> pd.DataFrame :
    """
    Eliminates rows for which the target column has no value.
    """
    print(f"\n[Drop line without value for {target_column.value}]")
    before = len(df)
    df.drop(df[df[target_column.value].isna()].index, inplace = True)
    after = len(df)
    print(f"\t {before-after} lines dropped, {after} remaining ({round(after/before*100, 2)} %)")
    return df


def target_column_binarization(df : pd.DataFrame, target_column : Target_Column, min : int, max: int) -> pd.DataFrame :
    """
    Binarization of the scores of the columns designated as target. We go from a score
    between 0 and 100 to two classes 0 and 1. The thresholds can be adjusted but it
    is recommended to focus on the "extreme" values which are considered more significant.
    To determine the scores you can use the method plot_answers_distribution().
    Warning : the values which are out of the thresholds pass in NaN then are eliminated.
    """
    print(f"\n[{target_column.value} binarization]")
    len_before = len(df[target_column.value].dropna())
    df[target_column.value] = [0 if x<=min else (1 if x>=max else np.NaN) for x in df[target_column.value]]    
    report = pd.concat([df[target_column.value].value_counts(dropna=True), pd.Series([len_before-len(df[target_column.value].dropna())], index=[np.NaN])])
    for i in report.index :
        print(f"\t {i} : {report[i]}")
    df = drop_lines_without_value_for_target_column(df=df, target_column=target_column)
    return df


def drop_columns_that_are_the_cause_of_incomplete_lines(df : pd.DataFrame, target_column : Target_Column) -> pd.DataFrame :
    """
    Drop the columns that have no values in front of the target column and display
    the proportion of missing values for each if any.
    """
    print("\n[Drop columns that are the cause of incomplete lines]")
    columns_to_drop = []
    tips = []
    for col in df.columns :
        temp = pd.concat([df[col], df[target_column.value]], axis=1)
        incomplete_line_count = len(temp) - len(temp.dropna())
        if incomplete_line_count > 0 :
            if incomplete_line_count == len(df[target_column.value]) :
                columns_to_drop.append(col)
            else : 
                tips.append((round((incomplete_line_count / len(df[target_column.value])) * 100, 2), col))
    for col in columns_to_drop :
        print("\t 100.00 % missing data in", col, "-> column droped")
    for tpl in tips :
        print("\t", tpl[0], "% missing data in", tpl[1])
    df = df.drop(columns_to_drop, axis=1)
    return df


def drop_incompletes_lines(df : pd.DataFrame) -> pd.DataFrame :
    """
    Drop the lines in which a value is missing.
    """
    print("\n[Drop incomplete lines]")
    with_nan_len = len(df)
    df.dropna(inplace=True)
    print(f"\t {with_nan_len - len(df)} incomplete lines deleted, {len(df)} remaining ({round(len(df) / with_nan_len * 100, 2)} %)")
    return df


def get_X_y_groups_timestamps(target_column : Target_Column, minimum_threshold_for_target_binarization : int, maximum_threshold_for_target_binarization : int) -> tuple[pd.DataFrame, pd.Series, pd.Series] :
    """
    Loads the original dataframe, cleans it, binarizes the y's and
    returns X, y and the groups (a group is in fact a participant).
    
    Examples
    --------
    >>> X, y, groups, timestamps = get_X_y_groups_timestamps(
            target_column=Target_Column.Morning_and_afternoon_stress_combined,
            minimum_threshold_for_target_binarization=20,
            maximum_threshold_for_target_binarization=60
        )
    """
    df = get_original_dataframe()
    df = eliminate_participants_for_whom_we_do_not_have_sufficient_data(original_df=df, minimum_number_of_lines_to_have_per_participant=24)
    df = clean_index(original_df=df)
    if target_column == Target_Column.Morning_and_afternoon_stress_combined :
        df = build_new_combined_Stress_column(df=df)
    df = drop_unusable_colums(df=df, target_column=target_column)
    df = remove_columns_with_infinite_values(df=df)
    df = drop_lines_without_value_for_target_column(df=df, target_column=target_column)
    df = target_column_binarization(df=df, target_column=target_column, min=minimum_threshold_for_target_binarization, max=maximum_threshold_for_target_binarization)
    df = drop_columns_that_are_the_cause_of_incomplete_lines(df=df, target_column=target_column)
    df = drop_incompletes_lines(df=df)
    X = df.drop([target_column.value, "UUID", "Timestamp"], axis=1)
    y = df[target_column.value]
    groups = df["UUID"]
    timestamps = df["Timestamp"]
    return X, y, groups, timestamps

