import os
import pandas as pd
from datetime import timedelta
import numpy as np



import my_paths



"""
all raw columns :
    'numer_sta', 'date', 'pmer', 'tend', 'cod_tend', 'dd', 'ff', 't', 'td', 'u',   
    'vv', 'ww', 'w1', 'w2', 'n', 'nbas', 'hbas', 'cl', 'cm', 'ch', 'pres', 
    'niv_bar', 'geop', 'tend24', 'tn12', 'tn24', 'tx12', 'tx24', 'tminsol',
    'sw', 'tw', 'raf10', 'rafper', 'per', 'etat_sol', 'ht_neige', 'ssfrai',
    'perssfrai', 'rr1', 'rr3', 'rr6', 'rr12', 'rr24', 'phenspe1',
    'phenspe2', 'phenspe3', 'phenspe4', 'nnuage1', 'ctype1', 'hnuage1',    
    'nnuage2', 'ctype2', 'hnuage2', 'nnuage3', 'ctype3', 'hnuage3',
    'nnuage4', 'ctype4', 'hnuage4' , 'Unnamed: 59'

to keep :
    'date'
    'dd' = direction du vent moyen 10mn (degré)
    'ff' = vitesse du vent moyen 10mn (m/s)
    't' = température (K)
    'u' = humidité (%)
    'vv' = visibilité horizontale (mètre)
    'ww' = temps présent (code (4677))
    'n' = nébulosité totale (%)
    'pres' = pression station (Pa)
    'rr12', 'rr24' = Précipitations dans les N dernières heures (mm)

types_dict = {
    'numer_sta':str, 'date':str , 'pmer':np.int64, 'tend':np.int64, 'cod_tend':np.int64, 'dd':np.int64, 'ff':np.float64, 't':np.float64, 'td':np.float64, 'u':np.int64,   
    'vv':np.float64, 'ww':np.int64, 'w1':np.int64, 'w2':np.int64, 'n':np.float64, 'nbas':np.int64, 'hbas':np.int64, 'cl':np.int64, 'cm':np.int64, 'ch':np.int64, 'pres':np.int64, 
    'niv_bar':np.int64, 'geop':np.int64, 'tend24':np.int64, 'tn12':np.float64, 'tn24':np.float64, 'tx12':np.float64, 'tx24':np.float64, 'tminsol':np.float64,
    'sw':int, 'tw':np.float64, 'raf10':np.float64, 'rafper':np.float64, 'per':np.float64, 'etat_sol':np.int64, 'ht_neige':np.float64, 'ssfrai':np.float64,
    'perssfrai':np.float64, 'rr1':np.float64, 'rr3':np.float64, 'rr6':np.float64, 'rr12':np.float64, 'rr24':np.float64, 'phenspe1':np.float64,
    'phenspe2':np.float64, 'phenspe3':np.float64, 'phenspe4':np.float64, 'nnuage1':np.int64, 'ctype1':np.int64, 'hnuage1':np.int64,    
    'nnuage2':np.int64, 'ctype2':np.int64, 'hnuage2':np.int64, 'nnuage3':np.int64, 'ctype3':np.int64, 'hnuage3':np.int64,
    'nnuage4':np.int64, 'ctype4':np.int64, 'hnuage4':np.int64, 'Unnamed: 59':np.int64
}
"""




def group_meteorological_data_for_Dijon_from_raw_csv_files(raw_weather_csv_folder_path : str, destination_directory_path : str) :
    """
    Method that recovers raw data from "Météo France" and transforms them to make them more exploitable.
    The result is a dataframe in the form of a parquet file and containing only the data concerning Dijon indexed by the time. 
    Note : dijon_longvic_station_code = 7280
    In addition, a rolling average is calculated for some of these indicators. This is the result of the difference between the
    average of the day and the average of the last 5 days. The purpose of this operation is to highlight changes in trends over time.
    
    + raw_weather_csv_folder_path : folder containing all the raw weather data we are interested in.
    They can be downloaded from : https://donneespubliques.meteofrance.fr/ >> Données libres d'accès >> Données SYNOP essentielles OMM
    + destination_directory_path : directory where we want the result file to be stored
    """
    # get raw CSV paths, read each CSV and load them in one big dataframe
    df = pd.concat([pd.read_csv(csv_path, sep=';') for csv_path in [os.path.join(raw_weather_csv_folder_path, item) for item in my_paths.my_listdir(directory_path=raw_weather_csv_folder_path, ignore_hidden_files=True, sort_by_creation_time=False)]])
    # replace "mq" values by Nan
    df.replace("mq", np.NaN, inplace=True)    
    # keep only data concerning DIJON
    df = df.loc[df["numer_sta"] == 7280]
    # keep only used data
    df = df[['date', 'dd', 'ff', 't', 'u', 'vv', 'ww', 'n', 'pres', 'rr12', 'rr24']]
    # set "date" column as index, convert index to pd.datetime, switch from UTC dates to French/Paris dates and sort index just to be sure
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M%S', utc=True)
    df.index = df.index.tz_convert('Europe/Paris')
    df.sort_index(inplace=True)
    # add columns with "Nan" for each rolling average columns
    columns_for_which_a_rolling_average_should_be_calculated = ['dd', 'ff', 't', 'u', 'vv', 'ww', 'n', 'pres']
    rolling_average_columns = [item + "_RA" for item in columns_for_which_a_rolling_average_should_be_calculated]
    df = df.reindex(columns=sorted(df.columns.tolist() + rolling_average_columns))
    # check column types
    df = df.astype({'dd':np.int64, 'ff':np.float64, 't':np.float64, 'u':np.int64, 'vv':np.float64, 'ww':np.int64, 'n':np.float64, 'pres':np.int64, 'rr12':np.float64, 'rr24':np.float64})
    # compute rolling average
    index_min = df.index.min().date()
    index_max = df.index.max().date()
    # (a) for each rolling average columns
    for i in range(len(columns_for_which_a_rolling_average_should_be_calculated)) :
        original_col = columns_for_which_a_rolling_average_should_be_calculated[i]
        ra_col = rolling_average_columns[i]
        # (b) for each day in the index
        for i in range((index_max - index_min).days + 1) :
            day_date = index_min + timedelta(days=i)
            # (c) compute day's average
            day_values = df.loc[str(day_date) , original_col]
            day_average = day_values.mean(skipna = True)
            # (d) compute last 5 days average when it's possible and attribute value
            if str(day_date - timedelta(days=5)) in df.index :
                last_5_days_values = df.loc[str((day_date - timedelta(days=5))) : str((day_date - timedelta(days=1))) , original_col] 
                last_5_days_average =  last_5_days_values.mean(skipna = True)
                df.loc[str(day_date), ra_col] = day_average - last_5_days_average
            else : 
               df.loc[str(day_date), ra_col] = np.NaN
    # end
    df.to_parquet(
        path=os.path.join(destination_directory_path, "dijon_weather_conditions.parquet.gzip"),
        compression='gzip'
    )
    print(df)
    




def demo() :

    raw_dataset_path = os.path.join(my_paths.downloads_directory_path, "MIAMS_Raw_Dataset")
    computed_dataset_path = os.path.join(my_paths.downloads_directory_path, "MIAMS_Computed_Dataset")

    group_meteorological_data_for_Dijon_from_raw_csv_files(
        raw_weather_csv_folder_path = my_paths.get_dataset_weather_directory_path(dataset_path=raw_dataset_path),
        destination_directory_path = my_paths.get_dataset_weather_directory_path(dataset_path=computed_dataset_path),
    )



    