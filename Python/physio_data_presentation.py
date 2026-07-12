import matplotlib.pyplot as plt
from pytz import NonExistentTimeError
from tqdm import tqdm
import os
import pandas as pd
from enum import Enum
import math
import numpy as np
import math
from matplotlib.backends.backend_pdf import PdfPages


import my_paths


class Recorded_Data(Enum):
    """
    Different types of data recorded by the Empatica bracelet
    """
    acc = {"csv" : "ACC.csv", "parquet" : "ACC.parquet.gzip", "freq" : 32, "note" : "Data from 3-axis acceleometer sensor in the range [-2g, 2g]. (sampled at 32 Hz)"}
    bvp = {"csv" : "BVP.csv", "parquet" : "BVP.parquet.gzip",  "freq" : 64, "note" : "Data from photoplethysmograph (PPG). (sampled at 64 Hz)"}
    eda = {"csv" : "EDA.csv", "parquet" : "EDA.parquet.gzip", "freq" : 4, "note" : "Data from the electrodermal activity sensor in μS. (sampled at 4 Hz)"}
    hr = {"csv" : "HR.csv", "parquet" : "HR.parquet.gzip", "freq" : 1, "note" : "This file contains the average heart rate values, computed in spans of 10 seconds"}
    ibi = {"csv" : "IBI.csv", "parquet" : "IBI.parquet.gzip", "freq" : None, "note" : "Inter beat intervals. (intermittent output with 1/64 second resolution)"}
    temp = {"csv" : "TEMP.csv", "parquet" : "TEMP.parquet.gzip", "freq" : 4, "note" : "Data from temperature sensor expressed in degrees on the Celsius (°C) scale (sampled at 4 Hz)"}


def my_super_reader(session_path : str, data : Recorded_Data) -> pd.DataFrame :
    """
    Reads the CSV file from any Empatica session as a Pandas dataframe.
    + session_path : Path to the directory of the empatica session
    + data : Type of physiological data to read
    > "ACC", "BVP", "EDA", "HR", "IBI", "TEMP"
    """
    # check if file is empty
    if os.path.getsize(session_path) == 0 :
        return pd.DataFrame()
    # infos to retrieve
    timestamp, df = None, None
    # ibi
    if data == Recorded_Data.ibi :
        with open(session_path, encoding='utf-8') as f:
            timestamp = float(str(f.readline()).strip().split(',')[0])
        df = pd.read_csv(session_path, skiprows=1, names=['ibi'], index_col=0)
        df['ibi'] *= 1000                                                                                   # to get ms
        df.index = pd.to_datetime((df.index * 1000 + timestamp * 1000).map(int), unit='ms', utc=True)
        df.index.name = 'datetime'
    # other
    else :
        with open(session_path, encoding='utf-8') as f:
            timestamp = pd.to_datetime(float(str(f.readline()).strip().split(',')[0]), unit='s', utc=True)
        if data == Recorded_Data.acc :
            df = pd.read_csv(session_path, skiprows=2, names=['acc_x', 'acc_y', 'acc_z'])
        if data == Recorded_Data.bvp :
            df = pd.read_csv(session_path, skiprows=2, names=['bvp'])
        if data == Recorded_Data.eda :
            df = pd.read_csv(session_path, skiprows=2, names=['eda'])
        if data == Recorded_Data.hr :
            df = pd.read_csv(session_path, skiprows=2, names=['hr'])
        if data == Recorded_Data.temp :
            df = pd.read_csv(session_path, skiprows=2, names=['temp'])
        df.index = pd.date_range(start=timestamp, periods=len(df), freq=str(1 / data.value.get("freq") * 1000) + 'ms', name='datetime', tz='UTC')
    # end
    df.sort_index(inplace=True)
    df.index = df.index.tz_convert('Europe/Paris')
    return df


def compute_duration(raw_dataset_path : str) -> tuple[dict[str, int], dict[str, np.array]]:
    """
    This function calculates two things. The first one is the global duration of the physiological
    records for each type of data. The second one is the global duration of the recordings for each
    type but hour by hour over 24 hours.
    + raw_dataset_path : path to the MIAMS dataset containing the raw data
    """
    # counters
    global_signal_duration = {"acc" : 0, "bvp" : 0, "eda" : 0, "hr" : 0, "ibi" : 0, "temp" : 0}
    hour_by_hour_signal_duration = {"bvp" : np.zeros(25, dtype=int), "ibi" : np.zeros(25, dtype=int), "eda" : np.zeros(25, dtype=int), "cleaned_eda" : np.zeros(25, dtype=int)}
    # for each dataset's session
    for session in tqdm(my_paths.get_all_dataset_sessions(dataset_path=raw_dataset_path)) :
        session_path = session[0]
        # for each kind of physiological data
        for item in Recorded_Data :
            # read as pd.Dataframe
            csv_path = os.path.join(session_path, item.value.get("csv"))
            df = my_super_reader(session_path=csv_path, data=item)
            # compute duration
            if not df.empty :
                # (1) IBI
                if item == Recorded_Data.ibi :
                    # browse the dataframe hour by hour
                    ibi_temp = np.zeros(25, dtype=int)
                    ibi_mini_floor = df.index.min().floor(freq='H')
                    ibi_maxi = df.index.max()
                    ibi_nbr_of_full_hours = math.ceil((ibi_maxi - ibi_mini_floor) / pd.Timedelta(hours=1))
                    for i in range(ibi_nbr_of_full_hours) :
                        # compute hour by hour signal duration
                        ibi_temp[ibi_mini_floor.hour] += df.loc[str(ibi_mini_floor)[:13]].sum()                                             # df.loc[mini : mini + pd.Timedelta(hours=1)] /!\ loc is insclusive
                        ibi_mini_floor = ibi_mini_floor + pd.Timedelta(hours=1)
                    # convert to seconds (currently in ms)
                    ibi_temp = ibi_temp / 1000
                    # add to counters
                    hour_by_hour_signal_duration[item.name] = np.add(hour_by_hour_signal_duration[item.name], ibi_temp)
                    global_signal_duration[item.name] += ibi_temp.sum()
                # (2) other physiological data
                else :
                    # (a) total duration
                    global_signal_duration[item.name] += (df.index.max() - df.index.min()).total_seconds()          # len(df.index) / item.value.get("freq")
                    # (b) classic hour by hour
                    if item == Recorded_Data.bvp or item == Recorded_Data.eda :
                        temp = np.zeros(25, dtype=int)
                        mini = df.index.min()
                        try :
                            mini_ceil = mini.ceil(freq='H')
                        except NonExistentTimeError :
                            # in the case of the time change on March 27, 2022
                            mini_ceil = (mini + pd.Timedelta(hours=1)).ceil(freq='H')
                        maxi = df.index.max()
                        maxi_floor = maxi.floor(freq='H')
                        # before mini_ceil, i.e. between the beginning and next full hour (mini_ceil)
                        temp[mini.hour] += (mini_ceil - mini).total_seconds() 
                        # for all full hours between mini_ceil and maxi_floor
                        for i in range(int((maxi_floor - mini_ceil) / pd.Timedelta(hours=1))) :
                            temp[(mini_ceil + pd.Timedelta(hours=i)).hour] += 3600
                        # after maxi_floor, i.e. between end and the previous full hour (maxi_floor)
                        temp[maxi.hour] += (maxi - maxi_floor).total_seconds()
                        # final
                        hour_by_hour_signal_duration[item.name] += temp                    
                        # (c) cleaned_eda hour by hour
                        if item == Recorded_Data.eda :
                            cleaned_eda_temp = np.zeros(25, dtype=int)
                            # retrieve outlier value
                            outlier_value = df.loc[df['eda'] < 0.05]
                            if not outlier_value.empty :
                                # browse outliers values and count their number for each hour (if there is)
                                eda_mini_floor = mini.floor(freq='H')   
                                nbr_of_full_hours = math.ceil((maxi - eda_mini_floor) / pd.Timedelta(hours=1))
                                for i in range(nbr_of_full_hours) :
                                    try :
                                        cleaned_eda_temp[eda_mini_floor.hour] += len(outlier_value.loc[str(eda_mini_floor)[:13]])
                                    except KeyError :
                                        pass
                                    eda_mini_floor = eda_mini_floor + pd.Timedelta(hours=1)
                                # convert nbr of point into seconds
                                cleaned_eda_temp = np.round(cleaned_eda_temp / item.value["freq"]).astype(int)
                            # finally, cleaned_eda is equal to the normal eda minus the time with the aberrant values
                            hour_by_hour_signal_duration["cleaned_eda"] += (temp - cleaned_eda_temp)
    # convert duration in hours
    for key in global_signal_duration :
        global_signal_duration[key] = global_signal_duration[key] / 3600
    for key in hour_by_hour_signal_duration :
        hour_by_hour_signal_duration[key] = hour_by_hour_signal_duration[key] / 3600
    # end
    print(global_signal_duration)
    print(hour_by_hour_signal_duration)
    return (global_signal_duration, hour_by_hour_signal_duration)







def graph_1_global_signal_duration(global_signal_duration: dict[str, int]) -> plt.figure :
    # editable parameters
    fig_title = "Total duration in hour for each type of recorded physiological data"
    x_label = "Type of recorded physiological data"
    y_label = "Duration in hour"
    choosed_cmap = plt.get_cmap("tab10")
    # plot
    fig = plt.figure(fig_title)
    barcontainer = plt.bar(*zip(*global_signal_duration.items()), color=choosed_cmap.colors)
    for rect_index, rect in enumerate(barcontainer):
        x = rect.get_x() + (rect.get_width() / 2.0)
        y = rect.get_height()
        plt.text(x, y, str(round(y)) + "h", ha='center', va='bottom')
    plt.title(fig_title, fontweight="bold")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    # end
    return fig


def graph_2_IBI_distribution(hour_by_hour_signal_duration : dict[str, np.array]) -> plt.figure :
    # editable parameters
    fig_title = "Average distribution of IBI and BVP data over 24 hours"
    x_label = "Time"
    y_label = "Duration in hour"
    # labels preparation 
    # labels_distribution = [datetime.time(hour=i, minute=j, second=0) for i in range(24) for j in range(60)]
    dist_labels = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "23:59", ]
    dist_x = np.arange(len(dist_labels))
    # plot
    fig_title = fig_title
    fig, ax = plt.subplots()
    bvp_barcontainer = plt.bar(x=dist_x + 0.5, height=hour_by_hour_signal_duration["bvp"], width=1.0, edgecolor = 'white', label="BVP")
    ibi_barcontainer = plt.bar(x=dist_x + 0.5, height=hour_by_hour_signal_duration["ibi"], width=1.0, edgecolor = 'white', label="IBI")
    average_ratio = 0
    for rect_index, rect in enumerate(ibi_barcontainer[:-1]):
        x = rect.get_x() + (rect.get_width() / 2.0)
        y = rect.get_height()
        ratio = (y / bvp_barcontainer[rect_index].get_height()) * 100
        average_ratio += ratio
        plt.text(x, y/2, " " + str(round(ratio)) + "%", ha='center', va='bottom', color="black")
    average_ratio = round(average_ratio / (len(ibi_barcontainer) -1))
    plt.title(fig_title, fontweight="bold")
    plt.xlabel(x_label)
    ax.set_xticks(dist_x, dist_labels)
    plt.xticks(rotation=90)
    plt.ylabel(y_label)
    plt.legend(title="Mean ratio : " + str(average_ratio) + " %")
    plt.show()
    # end
    return fig


def graph_3_EDA_distribution(hour_by_hour_signal_duration : dict[str, np.array]) -> plt.figure :
    # editable parameters
    fig_title = "Average distribution of EDA and Cleaned EDA data over 24 hours"
    x_label = "Time"
    y_label = "Duration in hour"
    # labels preparation 
    # labels_distribution = [datetime.time(hour=i, minute=j, second=0) for i in range(24) for j in range(60)]
    dist_labels = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "23:59", ]
    dist_x = np.arange(len(dist_labels))
    # plot
    fig_title = fig_title
    fig, ax = plt.subplots()    
    eda_barcontainer = plt.bar(x=dist_x + 0.5, height=hour_by_hour_signal_duration["eda"], width=1.0, edgecolor = 'white', label="EDA")
    cleaned_eda_barcontainer = plt.bar(x=dist_x + 0.5, height=hour_by_hour_signal_duration["cleaned_eda"], width=1.0, edgecolor = 'white', label="Cleaned EDA", color="limegreen")
    average_ratio = 0
    for rect_index, rect in enumerate(cleaned_eda_barcontainer[:-1]):
        x = rect.get_x() + (rect.get_width() / 2.0)
        y = rect.get_height()
        ratio = (y / eda_barcontainer[rect_index].get_height()) * 100
        average_ratio += ratio
        plt.text(x, y/2, " " + str(round(ratio)) + "%", ha='center', va='bottom', color="black")
    average_ratio = round(average_ratio / (len(cleaned_eda_barcontainer) -1))
    plt.title(fig_title, fontweight="bold")
    plt.xlabel(x_label)
    ax.set_xticks(dist_x, dist_labels)
    plt.xticks(rotation=90)
    plt.ylabel(y_label)
    plt.legend(title="Mean ratio : " + str(average_ratio) + " %")
    plt.show()
    # end
    return fig


def make_pdf_from_figures(figures : list[plt.figure], destination_dir : str, pdf_name : str):
    """
    Generate a pdf containing all the figures passed in parameter.
    + figures : list of figures to be included in the pdf
    + destination_dir : destination directory for the pdf
    + pdf_name : pdf name WITHOUT THE EXTENSION
    """
    # check parameters
    if len(figures) < 1:
        print("Error, figures list is empty...")
        return
    if not os.path.exists(destination_dir):
        print("Error, " + destination_dir + " doesn't exist...")
        return
    # make pdf
    pdf_path = os.path.join(destination_dir, pdf_name + ".pdf")
    pp = PdfPages(pdf_path)
    for f in figures:
        pp.savefig(f)
    pp.close()



def demo() :
    """
    As the raw dataset physiological data are stored in many CSV
    files, reading all these files takes a lot of time. That is why
    the demo function uses the pre-calculated values.
    """
    global_signal_duration = {'acc': 47534.912378472225, 'bvp': 47532.943307291665, 'eda': 47538.79152777778, 'hr': 47520.65833333333, 'ibi': 18989.05082, 'temp': 47496.97493055555}

    hour_by_hour_signal_duration = {
            'bvp': np.array([1931.69972222, 1992.17638889, 1972.75305556, 2012.8975    ,
            2017.47      , 2015.1475    , 2007.06111111, 1994.69194444,
            1984.16888889, 1992.48611111, 1975.77527778, 1951.19444444,
            1993.2225    , 2087.04      , 2122.38666667, 2111.28472222,
            2104.54027778, 2105.19      , 2050.72611111, 1977.21361111,
            1886.45916667, 1786.72277778, 1785.24472222, 1847.98527778,
                0.        ]),
            'ibi': np.array([1072.02015861, 1265.10424472, 1342.59499556, 1436.07801806,
            1453.81821083, 1439.86607444, 1346.96436778, 1159.07962917,
            976.33128722,  743.59511167,  629.30771083,  486.70011472,
            387.30939111,  474.57614861,  474.39211306,  462.19972278,
            441.12258861,  429.82404139,  405.62587833,  360.67133111,
            358.19676972,  437.21496944,  583.38491556,  823.07302667,
                0.        ]),
            'eda': np.array([1931.825     , 1992.38694444, 1972.77166667, 2012.97388889,
            2017.46416667, 2015.15861111, 2007.28277778, 1994.78916667,
            1984.31027778, 1992.60027778, 1975.99222222, 1951.38472222,
            1993.43416667, 2087.34277778, 2122.48916667, 2111.36972222,
            2104.65555556, 2105.51833333, 2051.03361111, 1977.81555556,
            1887.15805556, 1787.32027778, 1785.84638889, 1848.45833333,
                0.        ]),
            'cleaned_eda': np.array([1832.45361111, 1912.75777778, 1909.12972222, 1957.4575    ,
            1956.26361111, 1942.39222222, 1931.88361111, 1894.78944444,
            1859.61388889, 1836.005     , 1807.05861111, 1760.98305556,
            1792.56361111, 1897.15055556, 1948.00888889, 1945.22722222,
            1936.41972222, 1933.4       , 1872.35444444, 1776.96055556,
            1676.76222222, 1589.74722222, 1618.08583333, 1720.4575    ,
                0.        ])
        }

    fig_lst = []
    fig_lst.append(graph_1_global_signal_duration(global_signal_duration=global_signal_duration))
    fig_lst.append(graph_2_IBI_distribution(hour_by_hour_signal_duration=hour_by_hour_signal_duration))
    fig_lst.append(graph_3_EDA_distribution(hour_by_hour_signal_duration=hour_by_hour_signal_duration))
    make_pdf_from_figures(figures=fig_lst, destination_dir=my_paths.downloads_directory_path, pdf_name="Donnees_physio_presentation")


