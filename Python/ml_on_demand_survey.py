import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random



import my_paths
from ml_get_X_y_groups_timestamps import *


"""
This file contains the methods to read and obtain the data for the on demand surveys.
"""



def read_on_demand_dataframe() -> pd.DataFrame :
    """
    Read the on demand survey dataframe.
    """
    df = pd.read_parquet(my_paths.on_demand_survey_dataframe_path)
    return df


def transform_on_demand_survey_df_multiindex_to_fit_the_full_hours(on_demand_df : pd.DataFrame, original_df : pd.DataFrame ) -> pd.DataFrame :
    """
    Transform the multiindex of the on_demand_dataframe to fit the full hours original_df index.
    Basically the answers to the on_demand_survey are dated precisely (at the time
    of sending the form). In order to be able to merge them to the original_df data, which
    are on full time slot, the datetime index of the on_demand_df must be rounded.
    Note, we favor the higher hour because in the global dataframe, the 9:00
    line for example concerns data from 8:00 to 9:00. 
    """
    print("\n[transform_on_demand_survey_df_multiindex_to_fit_the_full_hours]")
    # Build the new index for the on_demand_df
    on_demand_new_multiindex = []
    idx_without_values_in_original_df_and_therefore_we_cant_keep_in_on_demand_df = []
    for idx in on_demand_df.index :
        if (idx[0], idx[1].ceil('H')) in original_df.index and not (idx[0], idx[1].ceil('H')) in on_demand_new_multiindex :
            on_demand_new_multiindex.append((idx[0], idx[1].ceil('H')))
        elif (idx[0], idx[1].floor('H')) in original_df.index  and not (idx[0], idx[1].floor('H')) in on_demand_new_multiindex :
            on_demand_new_multiindex.append((idx[0], idx[1].floor('H')))
        else :
            idx_without_values_in_original_df_and_therefore_we_cant_keep_in_on_demand_df.append(idx)
    on_demand_new_multiindex = pd.MultiIndex.from_tuples(on_demand_new_multiindex, names=('UUID', 'Timestamp'))
    # Delete the rows of the dataframe that are no longer in the new index
    on_demand_df.drop(idx_without_values_in_original_df_and_therefore_we_cant_keep_in_on_demand_df, axis=0, inplace=True)
    print(f"\t {len(idx_without_values_in_original_df_and_therefore_we_cant_keep_in_on_demand_df)} on_demand_survey did not find an equivalent in the original dataframe")
    print(f"\t {len(on_demand_df)} remaining")
    # End, assign the new index to on_demand_df
    on_demand_df.index = on_demand_new_multiindex
    return on_demand_df


def on_demand_survey_distribution() :
    """
    Plot on demand surey distributions.
    """
    df = read_on_demand_dataframe()
    sns.displot(data=df, x="stress_score_at_this_moment", kde=True).set(title="Score distribution")
    plt.show()
    df.reset_index(inplace=True)
    df["Timestamp"] = [x.hour for x in df["Timestamp"]]
    sns.displot(data=df, x="Timestamp", kde=True).set(title="Times at which forms are sent during the day")
    plt.show()


def check_on_demand_survey_score_and_original_dataframe_data_correspondences() :
    """
    Since the on-demand forms are very accurate, they are in theory more reliable.
    The purpose of this function is therefore to check whether the scores assigned
    in the global dataframe are consistent with the scores of the on demand surveys.
    """
    # read global dataframe and get score during the day
    global_df = get_original_dataframe()["survey_answer"].loc[:, (Target_Column.Noon_Morning_Stress.value, Target_Column.Evening_Afternoone_Stress.value)]
    global_df[Target_Column.Noon_Morning_Stress.value].fillna(global_df[Target_Column.Evening_Afternoone_Stress.value], inplace=True)
    global_df.drop(Target_Column.Evening_Afternoone_Stress.value, axis=1, inplace=True)
    global_df.rename(columns={"survey_answer_Noon_Morning-Stress": "Stress_Score"}, inplace=True)
    global_df.dropna(inplace=True)
    # read on demand survey dataframe
    on_demand_df = read_on_demand_dataframe()
    # compare whether the stress score in the on demand survey matches that in the global
    missing_values_counter = 0
    corresponding_values_counter = 0
    inconsistent_values_counter = 0
    floor_counter = 0
    ceil_counter = 0
    inconsistent_values = []
    for idx in on_demand_df.index :
        on_demand_score = on_demand_df.loc[idx]["stress_score_at_this_moment"]
        global_score = None
        if (idx[0], idx[1].ceil('H')) in global_df.index :
            ceil_counter += 1
            global_score = global_df.loc[(idx[0], idx[1].ceil('H')), :]["Stress_Score"]
        elif (idx[0], idx[1].floor('H')) in global_df.index :
            floor_counter += 1
            global_score = global_df.loc[(idx[0], idx[1].floor('H')), :]["Stress_Score"]
        else :
            missing_values_counter += 1
        if global_score != None :
            if on_demand_score-30 <= global_score <= on_demand_score+30 :
                corresponding_values_counter += 1
            else :
                inconsistent_values_counter += 1
                inconsistent_values.append((global_score, on_demand_score))
    # report 
    print(f"on_demand_df lenght : {len(on_demand_df)}")
    print(f"with values in the physio dataframes : {ceil_counter+floor_counter} ({round((ceil_counter+floor_counter) / len(on_demand_df) * 100 , 2)} %)")
    print(f"\tceil : {ceil_counter} \n\tfloor : {floor_counter}")
    print(f"missing physio value in the global dataframe : {missing_values_counter} ({round(missing_values_counter / len(on_demand_df) * 100 , 2)} %)")
    print(f"consistent stress scores between the two dataframes : {corresponding_values_counter} ({round(corresponding_values_counter / (ceil_counter+floor_counter) * 100 , 2)} %)")
    print(f"inconsistent stress scores between the two dataframes : {inconsistent_values_counter} ({round(inconsistent_values_counter / (ceil_counter+floor_counter) * 100 , 2)} %)")
    print("  global_score / on_demand_score")
    for tpl in inconsistent_values :
        print(f"\t {tpl[0]} \t {tpl[1]}")


def get_X_y_groups_timestamps_especially_for_on_demand_survey(add_0_artificially : bool, min_binarization : int = 20, max_binarization : int = 60) -> tuple[pd.DataFrame, pd.Series, pd.Series] :
    """
    This method uses the scores from the on-demand surveys. Each is associated with the original_df data when possible.
    Since the majority of on-demand surveys indicate stress, there is a parameter to add non-stressful situations into the X and y.
    These added non-stressful situations are taken each time for the same participants and at approximately the
    same times of the day as the initial stressful situations.


    Examples
    --------
    >>> X, y, groups, timestamps = get_X_y_groups_timestamps_especially_for_on_demand_survey(
            add_0_artificially=True,
            min_binarization=20,
            max_binarization=60
        )
    """
    # get global dataframe
    original_df = get_original_dataframe()
    acc_columns = original_df.loc[:, "acc"].columns.to_list()
    insufficient_weather_data = ["weather_n", "weather_n_RA"]
    unusable_survey_answers = original_df.loc[:, "survey_answer"].columns.to_list()
    to_drop = acc_columns + insufficient_weather_data + unusable_survey_answers
    original_df.columns = original_df.columns.get_level_values(1)
    # we select a first dataframe containing all the exploitable rows without taking into
    # account if they have a corresponding stress score
    data_only = original_df.copy()    
    data_only.drop(to_drop, axis=1, inplace=True)
    data_only = remove_columns_with_infinite_values(df=data_only)
    data_only.dropna(inplace=True)
    # then we select a second dataframe, like the first one it contains the usable rows
    # but with an associated stress score
    # Note that this one is smaller than the first one because there can only be a stress
    # score from 8 to 20 hours. The data of the night, even exploitable, do not appear in this one unlike the first one.
    data_and_stress_score = original_df.copy()
    data_and_stress_score[Target_Column.Morning_and_afternoon_stress_combined.value] = np.NaN
    data_and_stress_score[Target_Column.Morning_and_afternoon_stress_combined.value].fillna(data_and_stress_score[Target_Column.Noon_Morning_Stress.value], inplace=True)
    data_and_stress_score[Target_Column.Morning_and_afternoon_stress_combined.value].fillna(data_and_stress_score[Target_Column.Evening_Afternoone_Stress.value], inplace=True)
    data_and_stress_score[Target_Column.Morning_and_afternoon_stress_combined.value] = [0 if x<=min_binarization else (1 if x>=max_binarization else np.NaN) for x in data_and_stress_score[Target_Column.Morning_and_afternoon_stress_combined.value]]
    data_and_stress_score.drop(to_drop, axis=1, inplace=True)
    data_and_stress_score = remove_columns_with_infinite_values(df=data_and_stress_score)
    data_and_stress_score.dropna(inplace=True)
    # get on demand dataframe    
    on_demand_df = read_on_demand_dataframe()
    on_demand_df = transform_on_demand_survey_df_multiindex_to_fit_the_full_hours(on_demand_df=on_demand_df, original_df=data_only)
    # merge both on index
    df = data_only.loc[on_demand_df.index, :]
    df["on_demand_Stress_Score"] = on_demand_df["stress_score_at_this_moment"]
    # binarization
    df["on_demand_Stress_Score"] = [0 if x<=min_binarization else (1 if x>=max_binarization else np.NaN) for x in df["on_demand_Stress_Score"]]
    df.dropna(inplace=True)
    print("\n[on_demand_Stress_Score distribution after binarization]")
    counts = df["on_demand_Stress_Score"].value_counts()
    for i in range(len(counts)) :
        print(f'\t {counts.index[i]} : {counts[counts.index[i]]} ({round(counts[counts.index[i]]/len(df["on_demand_Stress_Score"])*100, 2)} %)')
    print('\n')
    # add 0 artificially
    if add_0_artificially :
        # we start by seeing the proportion of 0 and 1 in the dataframe
        stressful_situations = df.query("on_demand_Stress_Score == 1")
        zen_situations = df.query("on_demand_Stress_Score == 0")
        index_of_added_0 = []
        # if it has way too many 1's compared to 0's
        if len(stressful_situations) > len(zen_situations)*1.2 :
            # we build up a reserve of unexploited Zen situations that we can add to.
            # Note that the stress score of these Zen situations will not be the same as
            # the stress score of the on-demand forms, but rather the stress score of the daily lunch and evening forms.
            reserve_of_zen_situations = data_and_stress_score.loc[data_and_stress_score[Target_Column.Morning_and_afternoon_stress_combined.value] == 0]
            # for a certain number of stressful situations we will try to find a non-stressful equivalent 
            for idx in stressful_situations.index[:-len(zen_situations)] :
                # we create a small list of equivalent situations by doing our best
                idx_equivalent = []
                for x in reserve_of_zen_situations.index : 
                    if idx[0] == x[0] and idx[1].hour == x[1].hour and idx[1].day != x[1].day : 
                        idx_equivalent.append(x)
                if len(idx_equivalent) == 0 :
                    try :
                        search_index = reserve_of_zen_situations.loc[idx[0], :].index
                        idx_equivalent.append((idx[0], search_index[search_index.get_indexer([idx[1]], method="nearest")[0]]))
                    except KeyError : 
                        print("No more zen situation available for", idx[0])
                # we choose an equivalent among the list generated and we add it to the list of lines to add
                if len(idx_equivalent) > 0 :
                    choice = random.choice(idx_equivalent)
                    index_of_added_0.append(choice)
            # make sure that the zen situations we will add will not create duplicates in the df index
            duplicates = []
            for x in index_of_added_0 : 
                if x in df.index :
                    duplicates.append(x)
            for x in duplicates :
                index_of_added_0.remove(x)
            # when we have done, we add all these lines to the dataframe
            added_0 = reserve_of_zen_situations.loc[index_of_added_0, :]
            df = pd.concat([df, added_0])
            df["on_demand_Stress_Score"].fillna(df[Target_Column.Morning_and_afternoon_stress_combined.value], inplace=True)
            df.drop(Target_Column.Morning_and_afternoon_stress_combined.value, axis=1, inplace=True)
            print("\n[The Stress_Score column after adding 0]")
            counts = df["on_demand_Stress_Score"].value_counts()
            for i in range(len(counts)) :
                print(f'\t {counts.index[i]} : {counts[counts.index[i]]} ({round(counts[counts.index[i]]/len(df["on_demand_Stress_Score"])*100, 2)} %)')
            print('\n')
    # end
    X = df.reset_index(drop=True).drop("on_demand_Stress_Score", axis=1)
    y = df["on_demand_Stress_Score"]
    groups = df.index.get_level_values(0).to_series()
    timestamps = df.index.get_level_values(1).to_series()
    return X, y, groups, timestamps



#X, y, groups, timestamps = get_X_y_groups_timestamps_especially_for_on_demand_survey(
#            add_0_artificially=True,
#            min_binarization=20,
#            max_binarization=60
#)