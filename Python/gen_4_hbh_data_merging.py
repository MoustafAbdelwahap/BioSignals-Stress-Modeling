import datetime
import os
import pandas as pd
import numpy as np
import tqdm
from enum import Enum


import my_paths
from computing_parameters import Computing_Parameters


#######################################################################
# Data merging : Physiological features, Survey answers, weather data #
# /!\ Only with pre-processed data (computed dataset)                 #
# #####################################################################




## Classes used to manage the survey answer scores and where to distribute them ##

class Answers_Columns(Enum):
    """
    Names of the different columns in the survey_answer dataframe.
    > Note : Each score corresponds to a time of day
    """
    # Morning form scores
    Morning_Shape = "Morning_Shape"
    Morning_Sickness = "Morning_Sickness"
    Morning_Sleep = "Morning_Sleep"
    # Afternoone form score
    Noon_Morning_Stress = "Noon_Morning-Stress" 
    # Evening form scores
    Evening_Afternoone_Stress = "Evening_Stress"
    Evening_Negative_Valency = "Evening_Negative_Valency"
    Evening_Positive_Valency = "Evening_Positive_Valency"


class Day_Section() :
    """
    Different parts of a day, their start hour, their end hour and the Answers_Columns that correspond.
    > Note: start and end times are included

    + start_hour : Day_Section's start hour
    + end_hour : Day_Section's end hour
    + answers_columns : 
    """
    def __init__(self, start_hour : int, end_hour : int, answers_columns : list[Answers_Columns]) -> None:
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.answers_columns = answers_columns


class Sections(Enum):
    """
    Different parts of a day.
    > Note: hours have been chosen arbitrarily
    """
    morning_section = Day_Section(
        start_hour=0, 
        end_hour=8, 
        answers_columns=[Answers_Columns.Morning_Shape, Answers_Columns.Morning_Sickness, Answers_Columns.Morning_Sleep]
    )
    afternoon_section = Day_Section(
        start_hour=9, 
        end_hour=12, 
        answers_columns=[Answers_Columns.Noon_Morning_Stress]
    )
    evening_section = Day_Section(
        start_hour=13, 
        end_hour=21,
        answers_columns=[Answers_Columns.Evening_Afternoone_Stress, Answers_Columns.Evening_Negative_Valency, Answers_Columns.Evening_Positive_Valency]
    )






## My final readers ##

def read_physiological_features(computed_dataset_path : str, computing_parameters : Computing_Parameters) -> pd.DataFrame :
    """
    Reads physiological features as a dataframe.

    + computed_dataset_path : Dataset with computed features and merged for all participants
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    return pd.read_parquet(my_paths.get_parquet_containing_all_physiological_features(computed_dataset_path=computed_dataset_path, computing_parameters=computing_parameters))


def read_survey_answers(computed_dataset_path : str) -> pd.DataFrame :
    """
    Reads survey answers data as a dataframe.

    + computed_dataset_path : Dataset with pre-processed data
    """
    # retrieve file path
    thierry_survey_answers_file_path = my_paths.get_csv_containing_all_the_survey_answers(computed_dataset_path=computed_dataset_path)
    # read CSV as Dataframe and specify that hours are french hours
    answers_df = pd.read_csv(thierry_survey_answers_file_path)
    answers_df['Answer_Date'] = pd.to_datetime(answers_df['Answer_Date']).dt.tz_localize('Europe/Paris')
    # set 'Answer_user_id' and 'Answer_Date' as multi index and pivot Question column as multiple columns
    answers_df = answers_df.pivot(index=['Answer_user_id', 'Answer_Date'], columns="Question_Group", values="Score")
    # merge the lines with the same date to have the scores of a day on the same line
    answers_df.index = pd.MultiIndex.from_arrays([answers_df.index.get_level_values(0), answers_df.index.get_level_values(1).floor("D")]) # answers_df.index.set_levels(answers_df.index.get_level_values(1).floor("D"), level=1, verify_integrity=False)
    answers_df = answers_df.groupby(by=answers_df.index.names).first()
    answers_df.replace(to_replace=[None], value=np.NaN, inplace=True)   
    # convert colomns into multiindex 
    answers_df.columns = pd.MultiIndex.from_product([["survey_answer"], answers_df.columns])
    # end
    return answers_df


def read_meteorological_data(computed_dataset_path : str) -> pd.DataFrame :
    """
    Reads weather data as a dataframe.

    + computed_dataset_path : Dataset with pre-processed data
    """
    # load
    df = pd.read_parquet(my_paths.get_parquet_containing_dijon_weather_conditions(computed_dataset_path=computed_dataset_path))
    # convert colomns into multiindex 
    df.columns = pd.MultiIndex.from_product([["weather"], df.columns])
    # end
    return df






## Methods to merge all of this ##

def which_columns_should_be_retrieved_in_the_response_dataframe_for_a_timestamp(tsp : pd.Timestamp) -> list[str]:
    """
    Returns the list of score columns that match the timestamp passed in parameter.
    > Note : look enum above

    + tsp : timestamp for which we want to know
    """
    # check each section
    for section in Sections:
        if section.value.start_hour <= tsp.hour <= section.value.end_hour :
            return [answer_column.value for answer_column in section.value.answers_columns]
    # if none of them match the timestamp then None is returned
    return None


def is_it_a_rest_day(uuid : int , d : datetime) -> bool:
    """
    Indicates whether or not a participant was working on a given day.
    > Note: Please note that this method only works for MIAMS data. That is to say from February 2022 to May 2022.

    + uuid : participant's uuid
    + d : date to analyze
    """
    # (1) week-end
    if d.weekday() > 4 :
        return True
    # (2) statutory holidays
    statutory_holidays =  {
        '1er janvier': datetime.date(2022, 1, 1),
        'Lundi de Pâques': datetime.date(2022, 4, 18),
        '1er mai': datetime.date(2022, 5, 1),
        '8 mai': datetime.date(2022, 5, 8),
        'Ascension': datetime.date(2022, 5, 26),
        'Lundi de Pentecôte': datetime.date(2022, 6, 6),
        '14 juillet': datetime.date(2022, 7, 14),
        'Assomption': datetime.date(2022, 8, 15),
        'Toussaint': datetime.date(2022, 11, 1),
        '11 novembre': datetime.date(2022, 11, 11),
        'Jour de Noël': datetime.date(2022, 12, 25)
    }
    if d in statutory_holidays :
        return True
    # (3) holidays for students of psychology of Dijon
    # > Note: For vacations, the start and end dates are included as part of the vacation.
    if uuid not in [81, 82, 83, 84] :
        winter_holidays = (datetime.date(2022, 2, 19), datetime.date(2022, 2, 27))
        spring_holidays = (datetime.date(2022, 4, 16), datetime.date(2022, 5, 1))
        if (winter_holidays[0] <= d <= winter_holidays[1]) or (spring_holidays[0] <= d <= spring_holidays[1]) :
            return True
    # end
    return False


def is_the_participant_taking_an_exam(uuid : int , tsp : pd.Timestamp) -> bool :
    """
    Indicates whether or not a participant was taking an exam at a given time.

    + uuid : participant's uuid
    + tsp : date to analyze
    """
    L1 = {86, 90, 91, 93, 97, 110, 113, 114, 122, 123, 124, 126, 130}
    L2 = {85, 89, 94, 95, 96, 100, 101, 102, 103, 104, 105, 106, 108, 109, 111, 112, 115, 117, 118, 119, 120, 121, 125, 127, 128, 129}
    L3 = {99}
    # L1: 11 mai (13h à 14h ;  15h à 16h; 17h à 19h (stat;)  12 mai (13h à 14h; 15h à 16h)
    if uuid in L1 :
        if pd.Timestamp(year=2022, month=5, day=11, hour=13, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=11, hour=14, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=11, hour=15, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=11, hour=16, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=11, hour=17, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=11, hour=19, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=12, hour=13, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=12, hour=14, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=12, hour=15, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=12, hour=16, tz="Europe/Paris") :
            return True
    # L2:  11 mai (8h à 9h ; 10h à 11h)  /  12 mai (8h à 9h ; 10h à 11h)
    elif uuid in L2 :
        if pd.Timestamp(year=2022, month=5, day=11, hour=8, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=11, hour=9, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=11, hour=10, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=11, hour=11, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=12, hour=8, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=12, hour=9, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=12, hour=10, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=12, hour=11, tz="Europe/Paris") :
            return True
    # L3 : 2 mai 16h à 17h * 3 mai: 9h à 10h / 11h à 12h / 13h à 14h (epreuve difficilie) / 15h à 16h * 4 mai: 8h à 9h  / 10hà11h
    elif uuid in L3 :
        if pd.Timestamp(year=2022, month=5, day=2, hour=16, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=2, hour=17, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=3, hour=9, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=3, hour=10, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=3, hour=11, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=3, hour=12, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=3, hour=13, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=3, hour=14, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=3, hour=15, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=3, hour=16, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=4, hour=8, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=4, hour=9, tz="Europe/Paris") :
            return True
        if pd.Timestamp(year=2022, month=5, day=4, hour=10, tz="Europe/Paris") < tsp <= pd.Timestamp(year=2022, month=5, day=4, hour=11, tz="Europe/Paris") :
            return True
    return False


def merge_all(computed_dataset_path : str, computing_parameters : Computing_Parameters):
    """
    Merges multiple dataframes together to form a single dataframe containing all the different outputs of analyzable data.

    + computed_dataset_path : Dataset with pre-processed data
    + computing_parameters : different parameters that can be chosen when calculating the features of physiological signals
    """
    # load data
    physiological_features_df = read_physiological_features(computed_dataset_path=computed_dataset_path, computing_parameters=computing_parameters)
    answers_df = read_survey_answers(computed_dataset_path=computed_dataset_path)
    weather_df = read_meteorological_data(computed_dataset_path=computed_dataset_path)
    # create result dataframe, we take as a basis the physiological_features_df
    date_infos_mltindex = pd.MultiIndex.from_product([["date_infos"], ["week_day", "rest_day", "end_of_year_exam"]])
    df = physiological_features_df.reindex(columns = date_infos_mltindex.to_list() + physiological_features_df.columns.tolist() + answers_df.columns.tolist() + weather_df.columns.to_list())
    # for each row in the final dataframe (the one containing the new empty columns)
    for user_uuid, tsp in tqdm.tqdm(df.index) :
        ## (a) retrieve survey answers ##
        # determine in which columns of the answers_df we should look for the values for this row. This row contains only the physiological data for the moment.
        answers_columns = which_columns_should_be_retrieved_in_the_response_dataframe_for_a_timestamp(tsp)
        if answers_columns :
            # complete the result_df with the correct value(s) from the answers_df (if they exist)
            try :
                df.loc[(user_uuid, tsp), ("survey_answer", answers_columns)] = answers_df.loc[(user_uuid, pd.Timestamp(tsp.date(), tz='Europe/Paris')), ("survey_answer", answers_columns)]
            except KeyError as e:
                pass # print(f"Error, no answer corresponding ({user_uuid}, {tsp})")
        ## (b) retrieve weather ##
        # find nearest index in weather_df
        nearest_index =  weather_df.index[weather_df.index.get_indexer([tsp] , method='nearest')[0]]
        # attribute value
        try :
            df.loc[(user_uuid, tsp), ("weather", slice(None))] =  weather_df.loc[nearest_index , ("weather", slice(None))]
        except KeyError as e:
            print(f"Error, no weather data corresponding ({user_uuid}, {tsp})")
        ## (c) retrieve day's infos ##
        week_day = tsp.weekday()                                              # Monday == 0 … Sunday == 6
        rest_day = is_it_a_rest_day(uuid=user_uuid, d=tsp.date())
        end_of_year_exam = is_the_participant_taking_an_exam(uuid=user_uuid, tsp=tsp)
        temp = pd.Series([week_day, rest_day, end_of_year_exam], index=date_infos_mltindex)
        df.loc[(user_uuid, tsp), ("date_infos", slice(None))] = temp
    # end
    df.to_parquet(path= os.path.join(computed_dataset_path, my_paths.get_final_parquet_name(computing_parameters=computing_parameters)), compression='gzip')
    df.to_csv(path_or_buf=os.path.join(computed_dataset_path, my_paths.get_final_parquet_name(computing_parameters=computing_parameters))[:-13] + ".csv")
    print(df)






def demo() :
    # retrieve paths
    computed_dataset_name = "MIAMS_Computed_Dataset"
    computed_dataset_path = os.path.join(my_paths.downloads_directory_path, computed_dataset_name)
    
    computing_parameters = Computing_Parameters(
        window_length = 3600,
        window_step_size = 3600,
        hrv_threshold = 0.1,
        hrv_clean_data = False,
        only_on_full_hour_slots = True,
    )

    merge_all(
        computed_dataset_path=computed_dataset_path,
        computing_parameters=computing_parameters
    )
