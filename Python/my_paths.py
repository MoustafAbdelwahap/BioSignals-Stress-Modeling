import os
from pathlib import Path

from computing_parameters import Computing_Parameters


## Different directories on a PC that can be useful ##
user_directory_path = str(Path.home()) 
downloads_directory_path = os.path.join(user_directory_path, "Downloads")

MIAMS_app_directory_path = os.getcwd()[:os.getcwd().index(os.sep+"MIAMS")+7]
exploitation_directory_path = os.path.join(MIAMS_app_directory_path, "Code/analysis")
logs_directory_path = os.path.join(exploitation_directory_path, "Logs")
dataframe_directory_path = os.path.join(exploitation_directory_path, "Dataframe")
scripts_directory_path = os.path.join(exploitation_directory_path, "Scripts_for_SGE")


## Custom methods ##

def my_listdir(directory_path : str, ignore_hidden_files : bool, sort_by_creation_time : bool = False) -> list[str]:
    """
    Reviews a folder and lists its contents
    + directory_path : folder to review
    + ignore_hidden_files : if true, hidden files will not be listed with others
    + sort_by_creation_time : if we want the paths sorted in ascending order of file or directory creation date
    """
    # retrieves files and folders and ignore hiddens files
    result = list()
    for x in os.listdir(directory_path):
        if ignore_hidden_files :
            if not x.startswith('.'):
                result.append(os.path.join(directory_path, x))
        else :
            result.append(os.path.join(directory_path, x))
    # sort if it's asked
    if sort_by_creation_time:
        result.sort( key=lambda x: os.path.getmtime(x))
    # end
    return result


def keep_folder_only(list_of_path : list[str]) -> list[str]:
    """
    Filters a list of paths that are passed in parameter and returns a sub-list that contains only the directories among them.

    + list_of_path : list of paths to be filtered

    >>> Example :
    list_containing_only_the_directories_of_the_list_passed_in_parameter = keep_folder_only(list_of_path= list_containing_files_and_directories)
    """
    temp = []
    for p in list_of_path :
        if os.path.isdir(p) :
            temp.append(p)
    return temp


def recursive_copy(original_directory_path : str, copy_directory_path : str):
    """
    Recursive method used in copy_folder_architecture method.
    """
    for item in os.listdir(original_directory_path) :
        original_sub_folder_path = os.path.join(original_directory_path, item)
        if os.path.isdir(original_sub_folder_path) :
            corresponding_copy_sub_folder_path = os.path.join(copy_directory_path, item)
            if not os.path.exists(corresponding_copy_sub_folder_path):
                os.makedirs(corresponding_copy_sub_folder_path)
            recursive_copy(original_directory_path=original_sub_folder_path, copy_directory_path=corresponding_copy_sub_folder_path)


def copy_folder_architecture(original_directory_path : str, target_directory_path : str):
    """
    This method recreate the architecture of a folder.
    By "architecture" we mean the tree structure and all its subfolders but not the files.
    + original_directory_path : folder whose architecture we want to copy
    + target_directory_path : target path
    
    """
    # destination_directory = original_directory_path[:original_directory_path.rindex(os.sep)]
    #copy_folder_path = os.path.join(destination_directory, copy_name)
    if not os.path.exists(target_directory_path):
        os.makedirs(target_directory_path)
    recursive_copy(original_directory_path=original_directory_path, copy_directory_path=target_directory_path)




## Arbitrarily chosen names for the directories that contain the participants' sessions and answers ##
sessions_directory_name = "sessions"

answers_directory_name = "answers"
csv_containing_all_the_survey_answers = "daily_forms_scores.csv"

weather_directory_name = "weather"
parquet_containing_dijon_weather_conditions = "dijon_weather_conditions.parquet.gzip"

on_demand_survey_dataframe_name = "on_demand_survey_dataframe"
on_demand_survey_dataframe_path = os.path.join(dataframe_directory_path, on_demand_survey_dataframe_name)



## To manage dataset folders path ##

def get_dataset_sessions_directory_path(dataset_path : str) -> str:
    """
    Returns the path to the directory containing the sessions of each participant from the dataset's path.
    + dataset_path : path of the dataset
    """
    dataset_sessions_directory_path = os.path.join(dataset_path, sessions_directory_name)
    return dataset_sessions_directory_path

def get_parquet_containing_all_physiological_features(computed_dataset_path : str, computing_parameters : Computing_Parameters):
    """
    Get the parquet file path containing the features of the physiological signals of all the participants.

    + computed_dataset_path : Dataset with features already computed. 
    i.e. the result of the method "compute_features_for_all_dataset_sessions" of "1_physiological_features_computing.py"
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    return os.path.join(get_dataset_sessions_directory_path(dataset_path=computed_dataset_path), get_all_user_features_parquet_name(computing_parameters=computing_parameters))


def get_dataset_answers_directory_path(dataset_path : str) -> str:
    """
    Returns the path to the directory containing the asnwers of each participant from the dataset's path.
    + dataset_path : path of the dataset
    """
    dataset_answers_directory_path = os.path.join(dataset_path, answers_directory_name)
    return dataset_answers_directory_path

def get_csv_containing_all_the_survey_answers(computed_dataset_path : str) :
    """
    Retrieves the path to the file containing the answers of all participants.

    + computed_dataset_path : Dataset with weather data already "processed". 
    i.e. with the answers to the forms already grouped together.
    """
    return os.path.join(get_dataset_answers_directory_path(dataset_path=computed_dataset_path), csv_containing_all_the_survey_answers)


def get_dataset_weather_directory_path(dataset_path : str) -> str:
    """
    Returns the path to the directory containing the meteorological data from the dataset's path.
    + dataset_path : path of the dataset
    """
    dataset_weather_directory_path = os.path.join(dataset_path, weather_directory_name)
    return dataset_weather_directory_path

def get_parquet_containing_dijon_weather_conditions(computed_dataset_path : str) :
    """
    Retrieves the path to the file containing the weather data of Dijon.

    + computed_dataset_path : Dataset with weather data already "processed". 
    i.e. the result of the method "get_meteorological_data_for_Dijon_from_raw_csv_files" of "3_meteorological_data_treatment .py"
    """
    return os.path.join(get_dataset_weather_directory_path(dataset_path=computed_dataset_path), parquet_containing_dijon_weather_conditions)




def get_list_of_participant_directories_containing_sessions(dataset_path : str) -> list[str]:
    """
    Get list of directories containing the sessions of each participant.

    + dataset_path : path to the dataset
    """
    sessions_folder_path = get_dataset_sessions_directory_path(dataset_path=dataset_path)
    participant_directories_containing_sessions = [os.path.join(sessions_folder_path, folder) for folder in my_listdir(directory_path=sessions_folder_path, ignore_hidden_files=True, sort_by_creation_time=False)]
    participant_directories_containing_sessions = keep_folder_only(list_of_path=participant_directories_containing_sessions)
    participant_directories_containing_sessions.sort(key=lambda x: int(x[x.rindex(os.sep):].split("_")[1]))
    return participant_directories_containing_sessions


def get_path_to_all_sessions_of_a_participant(participant_directory_containing_sessions : str) -> list[str]:
    """
    Returns a list of paths to all sessions of a participant.

    + participant_directory_containing_sessions : path of the participant's sessions directory
    """
    path_to_all_sessions_of_a_participant = my_listdir(
        directory_path=participant_directory_containing_sessions,
        ignore_hidden_files=True,
        sort_by_creation_time=False
    )
    path_to_all_sessions_of_a_participant = keep_folder_only(list_of_path=path_to_all_sessions_of_a_participant)
    path_to_all_sessions_of_a_participant.sort(key=lambda x: int(x[x.rindex(os.sep):].split("_")[1]))
    return path_to_all_sessions_of_a_participant
    

def get_all_dataset_sessions(dataset_path : str) -> list[tuple[str, str, str]]:
    """
    Returns a list of all session directories in a dataset.
    (session_path, participant, session_id)
    """
    result = []
    # for each participant
    for participant_sessions_directory_path in get_list_of_participant_directories_containing_sessions(dataset_path=dataset_path) :
        # for each participant's sessions
        for session_path in get_path_to_all_sessions_of_a_participant(participant_directory_containing_sessions=participant_sessions_directory_path) :
            participant = participant_sessions_directory_path[participant_sessions_directory_path.rindex(os.sep)+1:]
            session = session_path[session_path.rindex(os.sep)+1:]
            result.append([session_path, participant, session])
    return result




## To manage features (parquet files) in the computed dataset ##

def get_session_features_parquet_name(session_id : str, computing_parameters : Computing_Parameters):
    """
    Generates the normalized name for the session features parquet file.

    + session_name : like "session_ID"
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    # define result parquet file name
    parquet_name = f"features_{session_id}_wl_{computing_parameters.window_length}_wss_{computing_parameters.window_step_size}_hrv_threshold_{int(computing_parameters.hrv_threshold*100)}pct_hrv_clean_data_{computing_parameters.hrv_clean_data}.parquet.gzip"
    # specify if you have used the only_on_full_hour_slots option
    if computing_parameters.only_on_full_hour_slots :
        parquet_name = "hbh_" + parquet_name
    # end
    return parquet_name


def get_user_features_parquet_name(user_uuid : str, computing_parameters : Computing_Parameters):
    """
    Generates the normalized name for the user's features parquet file.

    + user_id : like "user_ID"
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    # define result parquet file name
    parquet_name = f"features_{user_uuid}_wl_{computing_parameters.window_length}_wss_{computing_parameters.window_step_size}_hrv_threshold_{int(computing_parameters.hrv_threshold*100)}pct_hrv_clean_data_{computing_parameters.hrv_clean_data}.parquet.gzip"
    # specify if you have used the only_on_full_hour_slots option
    if computing_parameters.only_on_full_hour_slots :
        parquet_name = "hbh_" + parquet_name
    # end
    return parquet_name


def get_all_user_features_parquet_name(computing_parameters : Computing_Parameters):
    """
    Generates the normalized name for the parquet file that will contain the features of all participants.

    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    # define result parquet file name
    parquet_name = f"all_participants_features_wl_{computing_parameters.window_length}_wss_{computing_parameters.window_step_size}_hrv_threshold_{int(computing_parameters.hrv_threshold*100)}pct_hrv_clean_data_{computing_parameters.hrv_clean_data}.parquet.gzip"
    # specify if you have used the only_on_full_hour_slots option
    if computing_parameters.only_on_full_hour_slots :
        parquet_name = "hbh_" + parquet_name
    # end
    return parquet_name


def get_final_parquet_name(computing_parameters : Computing_Parameters):
    """
    Generates the normalized name for the final parquet file that will contain all the features (physiological features, survey answers, weather, etc...).

    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    # define result parquet file name
    parquet_name = f"final_wl_{computing_parameters.window_length}_wss_{computing_parameters.window_step_size}_hrv_threshold_{int(computing_parameters.hrv_threshold*100)}pct_hrv_clean_data_{computing_parameters.hrv_clean_data}.parquet.gzip"
    # specify if you have used the only_on_full_hour_slots option
    if computing_parameters.only_on_full_hour_slots :
        parquet_name = "hbh_" + parquet_name
    # end
    return parquet_name