# This notebook is used to compute features (EDA, IBIS, ACC, Temp) during a specific sliding window
#AND we anticipate the French time change, during the night of March 27, 2022, we add one hour. When it was 2 am, we went directly to 3 am

import os
from sys import path
import pandas as pd
import flirt
from pytz import NonExistentTimeError
from tqdm import tqdm
import math
import numpy as np

from computing_parameters import Computing_Parameters  # it is another py file that enable us to intialize some paramters like window_length and cleaning
import my_paths

## sliding_window ##
def sliding_window(data,window_size,window_step):
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_arrayhttps://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array
    import tensorflow as tf

    #remove the first line of the sampling rate
    data.drop(data.index[:1], inplace=True)

    data.drop(data.index[:1], inplace=True)
    sequences=tf.keras.utils.timeseries_dataset_from_array(
        data,
        targets=None ,
        sequence_length=window_size, #window size
        sequence_stride=window_step, #window step (Period between successive output sequences) use this one
        #sampling_rate=, #window step (Period between individual timesteps within sequences)
        #batch_size=128,
        #shuffle=False,
        #seed=None,
        #start_index=None,
        #end_index=None
        )  
    for batch in sequences:
        inputs = batch
        return inputs


## My readers ##

def read_eda(eda_csv_file : str) -> pd.DataFrame:
    # check if the csv file is not empty
    if os.path.getsize(eda_csv_file) == 0 :
        return pd.DataFrame()
    else :
        eda = flirt.reader.empatica.read_eda_file_into_df(eda_csv_file)
        # change index time zone
        eda.index = eda.index.tz_convert('Europe/Paris')
        return eda


def read_ibis(ibis_csv_file : str) -> pd.DataFrame:
    # check if the csv file is not empty
    if os.path.getsize(ibis_csv_file) == 0 :
        return pd.DataFrame()
    else :
        ibis = flirt.reader.empatica.read_ibi_file_into_df(ibis_csv_file)
        # change index time zone
        ibis.index = ibis.index.tz_convert('Europe/Paris')
        return ibis


def read_acc(acc_csv_file : str) -> pd.DataFrame:
    # check if the csv file is not empty
    if os.path.getsize(acc_csv_file) == 0 :
        return pd.DataFrame()
    else :
        acc = flirt.reader.empatica.read_acc_file_into_df(acc_csv_file)
        # change index time zone
        acc.index = acc.index.tz_convert('Europe/Paris')
        return acc


def read_temp(temp_csv_file : str) -> pd.DataFrame:
    # check if the csv file is not empty
    if os.path.getsize(temp_csv_file) == 0 :
        return pd.DataFrame()
    else :
        temp = flirt.reader.empatica.read_temp_file_into_df(temp_csv_file)
        # change index time zone
        temp.index = temp.index.tz_convert('Europe/Paris')
        return temp




## My physiological signals computing methods ##

def compute_eda_features(eda : pd.DataFrame, window_length : int, window_step_size : int) -> pd.DataFrame:
    # check if the signal is long enough
    if (eda.index.max() - eda.index.min()).total_seconds() < window_length :
        return pd.DataFrame()
    else :
        '''
        # compute features
        eda_features = flirt.get_eda_features(
            data=eda.iloc[:, 0],
            window_length=window_length,
            window_step_size=window_step_size,
            data_frequency=4
        )
        '''
        eda_features= sliding_window(eda.iloc[:, 0],window_length,window_step_size)
        '''
        # convert columns single index in multi-index
        eda_features.columns = eda_features.columns.astype(str)
        eda_features.columns = pd.MultiIndex.from_product([["eda"], eda_features.columns])
        # end
        '''
        return eda_features


def compute_hrv_features(ibis : pd.DataFrame, window_length : int, window_step_size : int, hrv_threshold : float, hrv_clean_data : bool) -> pd.DataFrame:
    # check if the signal is long enough
    if (ibis.index.max() - ibis.index.min()).total_seconds() < window_length : 
        return pd.DataFrame()
    else :
        # compute features
        '''
        hrv_features = flirt.get_hrv_features(
            data=ibis.iloc[:, 0],
            window_length=window_length,
            window_step_size=window_step_size,
            domains=["td", "fd", "nl", "stat"], # time domain, frequency domain, non-linear features, and/or statistical features
            threshold=hrv_threshold,
            clean_data=hrv_clean_data
        )
        '''
        #hrv_features= ibis.iloc[:, 0]  #BVP
        hrv_features= sliding_window(ibis.iloc[:, 0],window_length,window_step_size)

        '''
        # add ln(RMSSD)
        if "hrv_rmssd" in  hrv_features.columns :
            hrv_features = hrv_features.reindex(columns=hrv_features.columns.to_list() + ["hrv_ln_rmssd"])
            hrv_features["hrv_ln_rmssd"] = [x if x == np.NaN else math.log(x) for x in hrv_features["hrv_rmssd"]]
        # convert columns single index in multi-index
        hrv_features.columns = pd.MultiIndex.from_product([["hrv"], hrv_features.columns])
        # end
        '''
        return hrv_features


def compute_acc_features(acc : pd.DataFrame, window_length : int, window_step_size : int) -> pd.DataFrame:
    # check if the signal is long enough
    if (acc.index.max() - acc.index.min()).total_seconds() < window_length :
        return pd.DataFrame()
    else :
        '''
        # compute features
        acc_features = flirt.get_acc_features(
            data=acc[:],
            window_length=window_length,
            window_step_size=window_step_size,
            data_frequency=32,
        )
        '''
        #acc_features = data=acc[:]
        acc_features= sliding_window(acc.iloc[:],window_length,window_step_size)

        '''
        # convert columns single index in multi-index
        acc_features.columns = pd.MultiIndex.from_product([["acc"], acc_features.columns])
        # end
        '''
        return acc_features


def compute_temp_features(temp : pd.DataFrame, window_length : int, window_step_size : int) -> pd.DataFrame:
    """
    + temp : 
    + window_length : 
    + window_step_size : 
    """
    # check if the signal is long enough
    if (temp.index.max() - temp.index.min()).total_seconds() < window_length :
        return pd.DataFrame()
    else : 
        '''
        # compute features
        temp_features = flirt.get_stat_features(
            data=temp,
            window_length=window_length,
            window_step_size=window_step_size,
            data_frequency=4,
            entropies=True
        )
        '''
        #temp_features=temp
        temp_features= sliding_window(temp,window_length,window_step_size)

        '''
        # convert columns single index in multi-index
        temp_features.columns = pd.MultiIndex.from_product([["temp"], temp_features.columns])
        # end
        '''
        return temp_features


def merge_different_features_calculated_for_a_session(features_dataframe_list : list[pd.DataFrame], window_step_size : int) -> pd.DataFrame:
    """
    Merges the calculated features for the different physiological signals of a session.
    + features_df : list of dataframes each containing the features of a type of physiological signal (EDA, HRV, ACC, TEMP)
    + window_step_size : window step size used in the features computing (used for the new index)
    """
    merged_features_df = pd.concat(features_dataframe_list, axis=1, sort=False)
    new_index = pd.date_range(
        start=merged_features_df.iloc[0].name.ceil('s'),
        end=merged_features_df.iloc[-1].name.floor('s'),
        freq=str(window_step_size)+'s',
        tz='Europe/Paris'
    )
    merged_features_df = merged_features_df.reindex(index=merged_features_df.index.union(new_index))
    merged_features_df.interpolate(method='time', inplace=True)
    merged_features_df = merged_features_df.reindex(new_index)
    # end
    return merged_features_df




## General computing methods ###

def truncate_time_indexed_dataframe_to_have_only_full_hour_slots(df : pd.DataFrame) -> pd.DataFrame:
    """
    Cut a dataframe to have values only on full hour slots. Return empty Dataframe if the signal is too short.
    
    + df : time-indexed dataframe
    
    >>> Example : 
                                            eda
        datetime
        2022-02-14 13:22:49+01:00         0.000000
        2022-02-14 13:22:49.250000+01:00  0.115355
        ...                                    ...
        2022-02-15 16:46:18.500000+01:00  0.015381
        2022-02-15 16:46:18.750000+01:00  0.014099

                                            eda
        datetime
        2022-02-14 14:00:00+01:00         0.455010
        2022-02-14 14:00:00.250000+01:00  0.456291
        ...                                    ...
        2022-02-15 15:59:59.750000+01:00  5.307559
        2022-02-15 16:00:00+01:00         5.297306

    """
    # anticipate the French time change, during the night of March 27, 2022, we add one hour. When it was 2 am, we went directly to 3 am
    # For example, if we have a session that starts at 1:30 am (user_61 session_1477612) : 
    # In normal time it would have given 2 am but since 2 am does not exist the night of March 27, 2022, we fall on an error
    try :
        new_min = df.index.min().ceil(freq='H')
    except NonExistentTimeError :
        # Example's continuation : session starts at 1:30 am, we add an hour, get 3:30 am and then use floor to get 3:00 am
        new_min = (df.index.min() + pd.Timedelta(hours=1)).floor(freq='H') 
    new_max = df.index.max().floor(freq='H')
    if (new_max-new_min) >= pd.Timedelta(hours=1):
        return df.truncate(before=new_min, after=new_max)
    else:
        return pd.DataFrame()


def compute_session_features(session_folder_path : str, result_directory_path : str, computing_parameters : Computing_Parameters):
    """
    Computes features for a session and stores the result as a parquet file in the specified destination directory.
    
    + session_folder_path : path of the folder containing the session data
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    # retrieve session ID
    session_name = session_folder_path[session_folder_path.rindex(os.sep)+1:]
    # list of computed features for each kind of physiological signal
    features_dataframe_list = list()
    ## (1) Compute features for each kind of physiological signal ##
    #   for EDA, HRV, ACC and TEMP :
    #       read CSV and truncate signal if the option has been activated
    #       if it's good, continue
    #           compute features
    #           if features have been computed, continue
    #               add features to the
    # eda
    eda = read_eda(os.path.join(session_folder_path, "EDA.csv"))
    if not eda.empty and computing_parameters.only_on_full_hour_slots :
        eda = truncate_time_indexed_dataframe_to_have_only_full_hour_slots(df=eda)
    if not eda.empty :
        eda_features = compute_eda_features(eda=eda, window_length=computing_parameters.window_length, window_step_size=computing_parameters.window_step_size)
        if not eda_features.empty :
            features_dataframe_list.append(eda_features)
    # ibis
    ibis = read_ibis(os.path.join(session_folder_path, "IBI.csv"))
    if not ibis.empty and computing_parameters.only_on_full_hour_slots :
        ibis = truncate_time_indexed_dataframe_to_have_only_full_hour_slots(df=ibis)
    if not ibis.empty :
        hrv_features = compute_hrv_features(ibis=ibis, window_length=computing_parameters.window_length, window_step_size=computing_parameters.window_step_size, hrv_threshold=computing_parameters.hrv_threshold, hrv_clean_data=computing_parameters.hrv_clean_data)
        if not hrv_features.empty :
            features_dataframe_list.append(hrv_features)
    # acc
    acc = read_acc(os.path.join(session_folder_path, "ACC.csv"))
    if not acc.empty and computing_parameters.only_on_full_hour_slots :
        acc = truncate_time_indexed_dataframe_to_have_only_full_hour_slots(df=acc)
    if not acc.empty :
        acc_features = compute_acc_features(acc=acc, window_length=computing_parameters.window_length, window_step_size=computing_parameters.window_step_size)
        if not acc_features.empty :
            features_dataframe_list.append(acc_features)
    # temp
    temp = read_temp(os.path.join(session_folder_path, "TEMP.csv"))
    if not temp.empty and computing_parameters.only_on_full_hour_slots :
        temp = truncate_time_indexed_dataframe_to_have_only_full_hour_slots(df=temp)
    if not temp.empty :
        temp_features = compute_temp_features(temp=temp, window_length=computing_parameters.window_length, window_step_size=computing_parameters.window_step_size)
        if not temp_features.empty :
            features_dataframe_list.append(temp_features)
    ## (2) Merge all in a global dataframe and store it in parquet file ##
    if len(features_dataframe_list) > 0 :
        # merge features in one dataframe
        merged_features_df = merge_different_features_calculated_for_a_session(features_dataframe_list=features_dataframe_list, window_step_size=computing_parameters.window_step_size)
        # Generates the normalized name for the session features parquet file
        parquet_name = my_paths.get_session_features_parquet_name(
            session_id=session_name,
            computing_parameters=computing_parameters,
        )
        
        # write parquet file in the destination folder
        merged_features_df.to_parquet(
            path=os.path.join(result_directory_path, parquet_name),
            compression='gzip'
        )


def compute_features_for_all_dataset_sessions(raw_dataset_path : str, destination_dataset_path : str, computing_parameters : Computing_Parameters):
    """
    Compute features for each participant's session and store them in one parquet file in the session's folder.

    + dataset_path : dataset path with all sessions unzipped
    + destination_dataset_path : Dataset where the computed features will be stored. It must have the same directory tree.
        It can be the same or uou can use : 
        my_paths.copy_folder_architecture(original_directory_path=raw_dataset_path, copy_name="Computed_Dataset")
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    # retrieve all participant sessions directories paths
    src_participant_directories_containing_sessions = my_paths.get_list_of_participant_directories_containing_sessions(dataset_path=raw_dataset_path)
    # retrieve the path of the directory containing the sessions in the destination dataset
    dest_sessions_directory = my_paths.get_dataset_sessions_directory_path(dataset_path=destination_dataset_path)
    # for each participant
    for src_participant_sessions_directory_path in src_participant_directories_containing_sessions :
        # for each participant's sessions
        for src_session_path in my_paths.get_path_to_all_sessions_of_a_participant(participant_directory_containing_sessions=src_participant_sessions_directory_path) :
            # print indicators
            participant = src_participant_sessions_directory_path[src_participant_sessions_directory_path.rindex(os.sep)+1:]
            session = src_session_path[src_session_path.rindex(os.sep)+1:]
            print("\n", participant, session)
            # retrieve destination folder for session's computed features
            features_destination = os.path.join(dest_sessions_directory, participant, session)
            print(features_destination)
            # computing
            compute_session_features(
                session_folder_path=src_session_path,
                result_directory_path=features_destination,
                computing_parameters=computing_parameters
            )
