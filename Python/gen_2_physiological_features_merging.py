import os
import pandas as pd
import numpy as np
import tqdm



import my_paths
from computing_parameters import Computing_Parameters



## Physiological features merging (per participant and then between all participants) ##

def merge_features_of_all_sessions_of_a_participant(participant_sessions_directory_path : str, computing_parameters : Computing_Parameters):
    """
    Merges features that have been calculated according to certain parameters for all sessions of a participant. 
    The features are contacted end-to-end and the final result is sorted in chronological order. 
    Duplicates in the index are not allowed. If duplicates appear when adding a session's dataframe, it is ignored.
    The result is stored in the participant's sessions directory.

    + participant_sessions_directory_path : path of the participant's sessions directory with features already computed for each session
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    
    """
    # ex : user_44
    user_uuid = participant_sessions_directory_path[participant_sessions_directory_path.rindex(os.sep)+1:]
    # result
    result_df = pd.DataFrame()
    # for each participant's session
    for session_path in  my_paths.get_path_to_all_sessions_of_a_participant(participant_directory_containing_sessions=participant_sessions_directory_path) :
        # ex : session_1450226
        session_id = session_path[session_path.rindex(os.sep) + 1 :]
        # retrieve features parquet path
        features_parquet_name = my_paths.get_session_features_parquet_name(
            session_id=session_id,
            computing_parameters=computing_parameters
        )
        features_parquet_path = os.path.join(session_path, features_parquet_name)
        # if the file exists
        if os.path.exists(features_parquet_path) :
            # read session's features
            features = pd.read_parquet(path=features_parquet_path)
            # concat with verify_integrity = True to check overlaps (forbids duplicates in index) 
            try :
                result_df = pd.concat([result_df, features], verify_integrity=True, sort=False)
            except ValueError as e:
                print("Error,", user_uuid, session_id, "will be ignored...\n\t", e) 
    if not result_df.empty :
        # sort by time
        result_df.sort_index(inplace=True)
        # store result as file
        result_parquet_name = my_paths.get_user_features_parquet_name(
            user_uuid=user_uuid,
            computing_parameters=computing_parameters
        )
        result_df.to_parquet(
            path=os.path.join(participant_sessions_directory_path, result_parquet_name),
            compression='gzip'
        )
    else :
        print("Error, no features for", user_uuid, "...")
   

def merge_features_by_participant_for_all(dataset_path : str, computing_parameters : Computing_Parameters):
    """
    Use merge_features_of_all_sessions_of_a_participant method for all participant in the dataset.
    
    + dataset_path : Dataset with features already computed for each session
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals

    """
    # for each participant
    for participant_sessions_directory_path in my_paths.get_list_of_participant_directories_containing_sessions(dataset_path=dataset_path) :
        print(participant_sessions_directory_path)
        merge_features_of_all_sessions_of_a_participant(
            participant_sessions_directory_path=participant_sessions_directory_path,
            computing_parameters=computing_parameters
        )


def merge_all_participants_features_in_a_dataframe(dataset_path : str, computing_parameters : Computing_Parameters):
    """
    Merges the features of all participants into a global dataframe.

    + dataset_path : Dataset with features already computed and merge for each participant
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals

    """
    # final result
    result_df = pd.DataFrame()
    # for each participant
    for participant_sessions_directory_path in my_paths.get_list_of_participant_directories_containing_sessions(dataset_path=dataset_path) :
        # example : user_44
        user_uuid = participant_sessions_directory_path[participant_sessions_directory_path.rindex(os.sep)+1:]
        # (int64), example : 44
        uuid = np.int64(user_uuid.split("_")[1])
        # retrieve features parquet path
        participant_features_parquet_name = my_paths.get_user_features_parquet_name(
            user_uuid=user_uuid,
            computing_parameters=computing_parameters
        )
        features_parquet_path = os.path.join(participant_sessions_directory_path, participant_features_parquet_name)
        # if the file exists
        if os.path.exists(features_parquet_path) :
            # read participant's features
            features = pd.read_parquet(path=features_parquet_path)
            # multiindexing
            features.index = pd.MultiIndex.from_product([[uuid], features.index], names=["UUID", "Timestamp"])
            # concat
            result_df = pd.concat([result_df, features])
    if not result_df.empty :
        # store result as file
        parquet_name = my_paths.get_all_user_features_parquet_name(
            computing_parameters=computing_parameters
        )
        result_df.to_parquet(
            path=os.path.join(my_paths.get_dataset_sessions_directory_path(dataset_path=dataset_path), parquet_name),
            compression='gzip'
            )
    else :
        print("Error, no user's features found in", dataset_path)






def demo(): 
    
    # retrieve paths
    raw_dataset_path = os.path.join(my_paths.downloads_directory_path, "MIAMS_Raw_Dataset")
    computed_dataset_name = "MIAMS_Computed_Dataset"
    computed_dataset_path = os.path.join(my_paths.downloads_directory_path, computed_dataset_name)

    computing_parameters = Computing_Parameters(
        window_length=3600, 
        window_step_size=3600,
        hrv_threshold=0.1,
        hrv_clean_data=False,
        only_on_full_hour_slots=True
    )


    # merge by participant
    merge_features_by_participant_for_all(
        dataset_path=computed_dataset_path,
        computing_parameters=computing_parameters
    )

    # merge all
    merge_all_participants_features_in_a_dataframe(
        dataset_path=computed_dataset_path,
        computing_parameters=computing_parameters
    )
