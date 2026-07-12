
from datetime import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


from my_paths import *
from db_access import *
from e4_access import *


## General utility functions ## 


def get_all_correspondences(correspondences_sn_code_path : str) -> list[dict]:
    """
    Retrieve all matches between survey_user uuid , research code and empatica serial number from csv.
    + correspondences_sn_code_path : path to the CSV file containing the correspondence between survey_user_uuid, search code and serial number of the empatica device for each participant
    --> return : lis[dict]
        'session_id': MIAMS session ID
        'survey_user_uuid': participant's id in the database
        'research_code': participant's research code which makes him anonymous
        'empatica_sn': serial number of the participant's empatica device
        'start_date': date the participant joins the data acquisition campaign
        'end_date': date the participant leaves the data acquisition campaign
        'comment': comment about the participant
    """
    # check parameters
    if not os.path.exists(correspondences_sn_code_path):
        print("Error,", correspondences_sn_code_path, "doesn't exist...")
        return False
    correspondences = list()
    # read csv
    with open(correspondences_sn_code_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # ignore first line
        csv_fisrt_line = reader.__next__()
        for row in reader:
            # break if the line is empty
            if(row[1] == ""):
                break
            else :
                temp = {
                    'session_id': row[0],
                    'survey_user_uuid': row[1],
                    'research_code': row[2],
                    'empatica_sn': row[3],
                    'start_date': row[5],
                    'end_date': row[6],
                    'comment': row[-1],
                }
                correspondences.append(temp)
    return correspondences


def get_correspondences_for_survey_user_uuids(correspondences_sn_code_path : str, survey_user_uuids: list[int]) -> list[dict]:
    """
    Retrieve matches between survey_user uuid , research code and empatica serial number from csv for the uuids passed in parameter.
    + survey_user_uuids : survey_user_uuid of interest
    """
    correspondences = get_all_correspondences(correspondences_sn_code_path)
    temp = list()
    for c in correspondences:
        if int(c.get("survey_user_uuid")) in survey_user_uuids:
            temp.append(c)
    return temp


def get_correspondences_from_to_survey_user_uuid(correspondences_sn_code_path : str, start_uuid : int, end_uuid : int) -> list[dict]:
    """
    Retrieve matches between survey_user uuid , research code and empatica serial number from csv for the uuids passed in parameter.
    + start_uuid <= uuid <= end_uuid
    """
    correspondences = get_all_correspondences(correspondences_sn_code_path)
    temp = list()
    for c in correspondences:
        if  start_uuid <= int(c.get("survey_user_uuid")) <= end_uuid:
            temp.append(c)
    return temp
    

def number_of_days_between_a_date_and_today(day : int, month : int, year : int):
    """
    Count the number of days between a given date and today's date. This function is useful to automatically calculate the x parameter of the last_x_days_report function.
    day, month, year : starting date for the count.
    --> return asbolute value
    """
    d0 = date(year, month, day)
    d1 = datetime.now().date()
    delta = (d1 - d0).days
    return abs(delta)


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




## Dataset Sanity Check ##


def ibi_signal_recordings_analysis_for_the_downloaded_sessions(empatica_study_id, dataset_path : str) -> list[plt.figure]:
    """
    Analysis of the quality of the IBI signal recordings for the downloaded sessions.
    + empatica_study_id : Study ID as int or String
    + dataset_path : path to the dataset directory
    """
    # check parameters
    if not os.path.exists(dataset_path):
        print("Error,", dataset_path ,"doesn't exist...")
        return
    # retrieve informations from all sessions in the study
    e4 = MyE4Client()
    study_sessions_list = e4.study_sessions_list(empatica_study_id)
    # retrieve the path to the directory of each participant sorted by date to have them in the right order
    participants_directories_path = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path)]
    participants_directories_path.sort(key=lambda x: os.path.getmtime(x))
    # collected data
    sessions = list()
    full_session_duration_per_session = list()
    ibi_measurement_per_session = list()
    users = list()
    full_session_duration_per_user = list()
    ibi_measurement_per_user = list()
    # for each participant
    for participant_directory_path in participants_directories_path:
        # get user
        participant_directory_name = participant_directory_path.split(os.sep)[-1]
        users.append(participant_directory_name)
        # get the paths to the participants' subdirectories for survey responses and empatica sessions
        # --> participant_directory = [ participant_sessions_directory + participant_answers_directory ]
        participant_sessions_directory_path = os.path.join(participant_directory_path, sessions_directory_name)
        participant_answers_directory_path = os.path.join(participant_directory_path, answers_directory_name)
        # collected data for this participant
        overall_sessions_duration_in_hours_for_the_participant = 0
        overall_IBI_measure_duration_in_hours_for_the_participant = 0
        # for each participant's session
        for session_name in os.listdir(participant_sessions_directory_path):
            # get session
            sessions.append(session_name)
            # retrieve session path
            session_path = os.path.join(participant_sessions_directory_path, session_name)
            # retrieve the duration of the session
            session_duration = 0
            for session in study_sessions_list:
                if str(session) == session_name:
                    session_duration = float(session.duration) / 3600
                    break
            # calculate the overall duration of the IBI measurement of the session
            session_ibi_duration = 0
            ibi_csv_file_path = os.path.join(session_path, "IBI.csv")
            ibi_csv_file_size = os.path.getsize(ibi_csv_file_path)
            if ibi_csv_file_size > 0:
                with open(ibi_csv_file_path, newline='', encoding='utf-8') as ibi_csv_file:
                    csv_reader = csv.reader(ibi_csv_file, delimiter=',')
                    header = next(csv_reader)
                    for row in csv_reader:
                        try:
                            session_ibi_duration += float(row[1])
                        except:
                            print("Error, participant", participant_directory_path, "session", session_name, "\n\t row =", row)
            session_ibi_duration /= 3600
            # per session
            full_session_duration_per_session.append(session_duration)
            ibi_measurement_per_session.append(session_ibi_duration)
            # per user
            overall_sessions_duration_in_hours_for_the_participant += session_duration
            overall_IBI_measure_duration_in_hours_for_the_participant += session_ibi_duration
        full_session_duration_per_user.append(overall_sessions_duration_in_hours_for_the_participant)
        ibi_measurement_per_user.append(overall_IBI_measure_duration_in_hours_for_the_participant)
    # calculation of average session time and average IBI record time
    average_session_time = 0
    for session_duration in full_session_duration_per_session:
        average_session_time += session_duration
    average_session_time /= len(full_session_duration_per_session)
    average_ibi_session_time = 0
    for ibi_duration in ibi_measurement_per_session:
        average_ibi_session_time += ibi_duration
    average_ibi_session_time /= len(ibi_measurement_per_session)
    # plot ibi ratio per user
    title = "Total duration of empatica sessions per participant and for each, the part of IBI recorded"
    figure1 = plt.figure(title)
    bar_session = plt.bar(users, full_session_duration_per_user, label="Overall duration of the participant's sessions")
    bar_ibi = plt.bar(users, ibi_measurement_per_user, label="Overall duration of the participant's IBI records")
    for rect_index, rect in enumerate(bar_session):
        x = rect.get_x() + (rect.get_width() / 2.0)
        y = rect.get_height()
        text = ""
        try:
            text = " " + str(int(( ibi_measurement_per_user[rect_index] / full_session_duration_per_user[rect_index]) * 100 )) + "%"
        except:
            text = " 0%"
        plt.text(x, y, text, ha='center', va='bottom', rotation=90)
    plt.xticks(rotation=70)
    plt.title(title, fontweight="bold")
    plt.xlabel("Participants")
    plt.ylabel("Empatica sessions duration in hours and IBI ratio")
    plt.legend()
    plt.show()
    # plot ibi ratio per session
    title = "Duration of all Empatica sessions in the study, as well as the duration of the IBI recording for each."
    figure2 = plt.figure(title)
    bar_session = plt.bar(sessions, full_session_duration_per_session, label="Session duration")
    bar_ibi = plt.bar(sessions, ibi_measurement_per_session, label="Session's IBI record duration")
    plt.tick_params(
        bottom=False,                               # ticks along the bottom edge are off
        labelbottom=False                           # labels along the bottom edge are off
    )
    plt.title(title, fontweight="bold")
    plt.xlabel("Empatica session")
    plt.ylabel("Duration in hours")
    plt.legend()
    plt.show()
    # plot average session time and average ibi mesurement time
    title = "Average duration of an Empatica session and average duration of the IBI recording ( ~" + str(int((average_ibi_session_time/average_session_time * 100))) + " % )"
    figure3 = plt.figure(title)
    bar_width = 0.3
    bar_session = plt.bar(0 - bar_width, average_session_time, width=bar_width, label="Average session time")
    x_text = bar_session.patches[0].get_x() + (bar_session.patches[0].get_width() * 0.33)
    y_text = bar_session.patches[0].get_height() + 0.1
    plt.text(x_text, y_text, str(timedelta(hours=average_session_time)).rsplit(':', 1)[0] + " h")
    bar_ibi = plt.bar(0 + bar_width, average_ibi_session_time, width=bar_width, label="Average IBI session time")
    x_text = bar_ibi.patches[0].get_x() + (bar_ibi.patches[0].get_width() * 0.33)
    y_text = bar_ibi.patches[0].get_height() + 0.1
    plt.text(x_text, y_text, str(timedelta(hours=average_ibi_session_time)).rsplit(':', 1)[0] + " h")
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.xlim((-1, 1))
    plt.xticks([])
    plt.ylabel("Average duration")  
    plt.show()
    # end
    return [figure1, figure2, figure3]




## Periodic report ## 


def number_of_survey_answers_per_survey_user_for_each_survey(correspondences : list[dict], report_analysis_start_date : datetime, display_values_on_the_chart_itself : bool = False, color_palette = plt.cm.tab10, bar_width : float = 0.3, estimated_number_of_answers_per_day :int = 3) -> plt.figure:
    """
    Show participant's responses for each survey since report_analysis_start_date
    + correspondences : correspondence between survey_user_uuid, search code and Empatica serial number for the participants you want to analyze
    + report_analysis_start_date : date from which data is measured for the report
    + display_values_on_the_chart_itself : indicates whether in addition to the table, you want to display the values in the bars of the graphs
    + color_palette : color palette for the chart
    + bar_width : bar width for the chart
    + estimated_number_of_answers_per_day : number of responses expected from a participant per day
    """
    # list of surveys available in the database
    database_surveys = get_survey(Queries.get_all_survey.value)
    database_surveys = sorted(database_surveys, key=lambda survey: survey.uuid)
    # collected data
    survey_user_uuid = np.zeros(len(correspondences), dtype=int)
    survey_answers_per_survey_user_uuid_for_each_survey = np.zeros((len(database_surveys), len(correspondences)), dtype=int)
    expected_number_of_answers_per_user  = np.zeros(len(correspondences), dtype=int)
    # for each selected participant
    for c_index, c in enumerate(correspondences):
        # get the main informations from the correspondence
        c_user_uiid = c.get("survey_user_uuid")
        c_start_date = datetime.strptime(c.get("start_date") + " 12:00:00", '%d/%m/%Y %H:%M:%S')
        c_end_date = datetime.strptime(c.get("end_date") + " 11:59:59", '%d/%m/%Y %H:%M:%S')
        ## (1) List of survey user uuid ##
        survey_user_uuid[c_index] = c_user_uiid
        ## (2) Participant's responses for each survey ##
        request = "SELECT * FROM survey_answer WHERE user_id = '"+ c_user_uiid +"' AND answer_date > '"+ str(report_analysis_start_date) +"'"
        all_user_answers = get_survey_answer(request)
        for answer in all_user_answers:
            for survey_index, survey in enumerate(database_surveys):
                if answer.survey_id == survey.uuid:
                    (survey_answers_per_survey_user_uuid_for_each_survey[survey_index])[c_index] += 1
                    break
        ## (3) Estimate of the time the participant should have uploaded ##
        number_of_day = 0
        if c_start_date <= report_analysis_start_date <= c_end_date:
            c_start_date = report_analysis_start_date
        if  c_start_date <= datetime.now() <= c_end_date :
            c_end_date = datetime.now()
        number_of_day = (c_end_date.date() - c_start_date.date()).days
        expected_number_of_answers_per_user[c_index] = number_of_day * estimated_number_of_answers_per_day
    # convert np.array to list of string (simpler for the continuation)
    survey_user_uuid = list(map(str, survey_user_uuid))
    # plot result 
    title = "Participant's responses for each survey since " + str(report_analysis_start_date)
    # for bar chart and the table
    f = plt.figure(title)
    row_labels = ["user uuid"]
    for s in database_surveys:
        row_labels.append(s.title)
    colors = ["pink"]
    for survey_index, s in enumerate(database_surveys):
        colors.append(color_palette(survey_index))
    table_cell_text = [survey_user_uuid]
    y_offset = np.zeros(len(survey_user_uuid))       # Initialize the vertical-offset for the stacked bar chart (so the next bar does not overlap the previous one)
    # plot graph
    plt.plot(survey_user_uuid, expected_number_of_answers_per_user, marker='_', linestyle="", alpha=0.8, color="black", label="expected number of survey_answers\n"+ str(estimated_number_of_answers_per_day) + " per day")
    for survey_index in range(len(database_surveys)):
        bar = plt.bar(survey_user_uuid, survey_answers_per_survey_user_uuid_for_each_survey[survey_index], bar_width, bottom=y_offset, color=color_palette(survey_index))
        if display_values_on_the_chart_itself:
            for rect_index, rect in enumerate(bar):
                if survey_answers_per_survey_user_uuid_for_each_survey[survey_index][rect_index] != 0:
                    x = rect.get_x() + (rect.get_width() / 2.0)
                    y = rect.get_y() + (rect.get_height() / 2.0) - 0.3
                    text = survey_answers_per_survey_user_uuid_for_each_survey[survey_index][rect_index]            
                    plt.text(x, y, text, ha='center', va='bottom')
        y_offset = y_offset + survey_answers_per_survey_user_uuid_for_each_survey[survey_index]
        table_cell_text.append(survey_answers_per_survey_user_uuid_for_each_survey[survey_index])
    # plot table
    table = plt.table(
        cellText=table_cell_text,
        cellLoc='center',
        rowLabels=row_labels,
        rowColours=colors,
        loc='bottom',
        bbox=[0.0, -0.4, 1, 0.3] # table position and size --> left, bottom, width, height
    )
    # final touches
    plt.xlim([-0.5, len(survey_user_uuid)-0.5])        # allows you to align the bars with the correct column in the table
    table.auto_set_font_size(False)                 # increase font size in table
    table.set_fontsize(10)
    plt.subplots_adjust(left=0.2, bottom=0.3)      # Adjust layout to make enought room for the table
    plt.tick_params(
        bottom=False,                               # ticks along the bottom edge are off
        labelbottom=False                           # labels along the bottom edge are off
    ) 
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.show()
    return f


def duration_of_empatica_sessions_uploaded_per_survey_user(empatica_study_id, correspondences : list[dict], report_analysis_start_date : datetime, estimated_number_of_hours_to_upload_per_day : float = 15.0 , color_palette = plt.cm.tab10, bar_width : float = 0.3) -> plt.figure:
    """
    Plots a bar chart showing the total duration of empatica recordings per user since report_analysis_start_date. Also indicates the number of sessions
    + empatica_study_id : study ID as int or String
    + correspondences : correspondence between survey_user_uuid, search code and Empatica serial number for the participants you want to analyze
    + report_analysis_start_date : date from which data is measured for the report
    + estimated_number_of_hours_to_upload_per_day : estimation of the number of hours to post per day to determine if a participant is late or not
    + color_palette : color palette for the chart
    + bar_width : bar width for the chart
    """
    # list of surveys available in the database
    database_surveys = get_survey(Queries.get_all_survey.value)
    database_surveys = sorted(database_surveys, key=lambda survey: survey.uuid)
    # collected data
    survey_user_uuid = np.zeros(len(correspondences), dtype=int)
    time_recorded_with_empatica_per_user_in_hours = np.zeros(len(correspondences), dtype=float)
    number_of_sessions_recorded_per_user = np.zeros(len(correspondences), dtype=int)
    expected_number_of_hours_per_user  = np.zeros(len(correspondences), dtype=float)
    # E4 client initialization and empatica informations retrieval
    e4 = MyE4Client()
    empatica_all_sessions = e4.study_sessions_list(study_id = empatica_study_id)
    empatica_all_devices = e4.study_device_usage_from_to_Dates(empatica_study_id, report_analysis_start_date, datetime.now())
    # for each selected participant    
    for c_index, c in enumerate(correspondences):
        # get the main informations from the correspondence
        c_user_uiid = c.get("survey_user_uuid")
        c_empatica_sn = c.get("empatica_sn")
        c_start_date = datetime.strptime(c.get("start_date") + " 12:00:00", '%d/%m/%Y %H:%M:%S')
        c_end_date = datetime.strptime(c.get("end_date") + " 11:59:59", '%d/%m/%Y %H:%M:%S')
        ## (1) List of survey user uuid ##
        survey_user_uuid[c_index] = c_user_uiid
        ## (2) Participant's empatica sessions and recorded time ##
        c_hardware_code = None
        for d in empatica_all_devices:
            if c_empatica_sn == d.serial_number:
                c_hardware_code = d.hardware_code
                break 
        for s in empatica_all_sessions:
            if s.device_id == c_hardware_code:
                if s.duration > 0 :
                    if c_start_date <= datetime.fromtimestamp(s.start_time) <= c_end_date :
                        if report_analysis_start_date <= datetime.fromtimestamp(s.start_time):
                            time_recorded_with_empatica_per_user_in_hours[c_index] += (s.duration / 3600)
                            number_of_sessions_recorded_per_user[c_index] += 1
        ## (3) Estimate of the time the participant should have uploaded ##
        number_of_day = 0
        if c_start_date <= report_analysis_start_date <= c_end_date:
            c_start_date = report_analysis_start_date
        if  c_start_date <= datetime.now() <= c_end_date :
            c_end_date = datetime.now()
        number_of_day = (c_end_date.date() - c_start_date.date()).days
        expected_number_of_hours_per_user[c_index] = number_of_day * estimated_number_of_hours_to_upload_per_day
    # plot result
    title = "Total duration of all recorded sessions per user since " + str(report_analysis_start_date)
    # transform "survey_user_uuid" into a list of string like ["user 1", "user 2", ...]
    temp = []
    for uuid in survey_user_uuid :
        temp.append("user " + str(uuid))
    survey_user_uuid = temp
    # for bar chart
    f = plt.figure(title)
    bar_width = 0.3
    color_palette = plt.cm.tab10
    # plot chart
    bar = plt.bar(survey_user_uuid, time_recorded_with_empatica_per_user_in_hours, bar_width, color=color_palette(0))
    plt.plot(survey_user_uuid, expected_number_of_hours_per_user, marker='_', linestyle="", alpha=0.8, color="r", label="expected number of hours\n"+ str(estimated_number_of_hours_to_upload_per_day) + " h per day")
    for rect_index, rect in enumerate(bar):
        x = rect.get_x() + (rect.get_width() / 2.0)
        y = rect.get_y() + (rect.get_height() / 2.0)
        text = "~ " + str(int(time_recorded_with_empatica_per_user_in_hours[rect_index])) + "h" # str(timedelta(hours=time_recorded_with_empatica_per_user_in_hours[rect_index])).rsplit(':', 1)[0] + " h"   # str(number_of_sessions_recorded_per_user[rect_index])
        plt.text(x, y, text, ha='center', va='bottom', rotation=90, color="white")
        y = rect.get_y() + rect.get_height()
        text =  str(number_of_sessions_recorded_per_user[rect_index])
        plt.text(x, y, text, ha='center', va='bottom', rotation=0)
        # x1 = rect.get_x() - (bar_width / 2.0)
        # x2 = rect.get_x() + bar_width * 1.5
        # plt.plot(x1, expected_number_of_hours_per_user[rect_index], marker=9, linestyle="", alpha=0.8, color="r")
        # plt.plot(x2, expected_number_of_hours_per_user[rect_index], marker=8, linestyle="", alpha=0.8, color="r")
    # final touches
    plt.subplots_adjust(left=0.2, bottom=0.1)      # Adjust layout to make enought room for the table
    plt.title(title, fontweight="bold")
    plt.xlabel("Participants")
    plt.xticks(rotation=70)
    plt.ylabel("Total duration of all participant's recorded sessions in hours and the number of sessions")
    plt.legend()
    plt.show()
    return f


def last_x_days_report(empatica_study_id, correspondences : list[dict], x : int = 0) -> list[plt.figure]:
    """
    Generates a report giving an overview of the situation over the last X days. You can select which participants will be taken into account.
    """
    # calculate start date of the analysis for the report
    today_5_am = datetime.now().replace(hour=5, minute=0, second=0, microsecond=0)
    report_analysis_start_date = today_5_am - timedelta(days = x)
    # measure the desired metrics using the appropriate functions
    figure1 = number_of_survey_answers_per_survey_user_for_each_survey(correspondences, report_analysis_start_date)
    figure2 = duration_of_empatica_sessions_uploaded_per_survey_user(empatica_study_id, correspondences, report_analysis_start_date)
    # end
    return [figure1, figure2]




## Check survey_answer values distribution ##


def extract_all_answer_values_in_survey_answer(answer_values : list, answer : dict):
    """
    Retrieve all answer values in an survey_answer.answer
    + answer_values : list in which all data retrieved from this survey_answer will be stored
    + answer : answer from which we wish to extract the values
    """
    if answer["type"] == "Group":
        data_field = answer["data"]     # type list 
        if type(data_field[0]) == str:
            for item in data_field:
                json_dict = json.loads(item)
                extract_all_answer_values_in_survey_answer(answer_values, json_dict)
        elif type(data_field[0]) == dict :
            for item in data_field:
                answer_values.append(item["data"])
    else :
        print("Error, we have something other than a 'Group'...")


def survey_answer_values_distribution(survey_uuid_to_exclude : list[int] = [10, 30, 31, 32]) -> list[plt.figure]:
    """
    Retrieves all the survey_answer values for the surveys that are not in the list of those to be excluded and a histogram is drawn to see their distribution for each survey
    + correspondences : correspondence between survey_user_uuid, search code and Empatica serial number for the participants you want to analyze
    + report_analysis_start_date : date from which data is measured for the report
    + survey_uuid_to_exclude : list of survey IDs to exclude
    # attention, this function returns a list of figures
    """
    # figures list
    figures_list = []
    # list of surveys available in the database
    database_surveys = get_survey(Queries.get_all_survey.value)
    database_surveys = sorted(database_surveys, key=lambda survey: survey.uuid)
    # for each kind of survey in the database
    for survey in database_surveys:
        # ignore this survey if it is on the list of those to be excluded
        if not survey.uuid in survey_uuid_to_exclude:
            # retrieve all the responses to this survey since the date and for users we are interested in and extract all the values
            request = "SELECT * FROM survey_answer WHERE survey_id = '" + str(survey.uuid) +"' AND user_id > '15'"
            survey_s_answers = get_survey_answer(request)
            all_answers_values = []
            for a in survey_s_answers:
                extract_all_answer_values_in_survey_answer(all_answers_values, a.answer.dictionnary)
            # filter values
            temp = list()
            print_legend = False # we display the legend only if non-empty strings are found in the answers
            for val in all_answers_values:
                try:
                    val = int(float(val))
                    temp.append(val)
                except ValueError:
                    if len(val.strip()) > 0:
                        temp.append(-10)
                        print_legend = True
            all_answers_values = temp
            # plot result
            title = "Responses distribution for the '" + survey.title + "' on " + str(len(survey_s_answers)) + " survey_answers"
            f = plt.figure(title)
            y, x, patches = plt.hist(
                x = all_answers_values,
                bins = 50,
                label = "the bar at x = -10 represents the number of string responses"
            )
            # a little bit of color
            color_map = plt.cm.get_cmap("RdYlGn")
            max_val = y.max() * 1.1 # the multiplication by 1.1 is only for the aesthetics of the colors
            for patch in patches:
                mix = 1 - (patch.get_height() / max_val)
                patch.set_facecolor(color_map(mix))
            # final touches
            plt.title(title, fontweight="bold")
            plt.xlabel("Answer types")
            plt.ylabel("Response frequencies")
            if print_legend:
                plt.legend()
            plt.show()
            figures_list.append(f)
    return figures_list




## Focus on some participants ##


def duration_of_empatica_sessions_uploaded_per_survey_user_and_by_time(study_id, correspondences : list[dict], show_points : bool = False):
    """
    Plots the evolution of the volume of data posted by participant as a function of time.
    + study_id : empatica corresponding study ID
    + correspondences : correspondence between survey_user_uuid, search code and Empatica serial number for the participants you want to analyze
    + show_points : show point on the chart
    """
    # retrieve all sessions and devices
    e4 = MyE4Client()
    all_sessions = e4.study_sessions_list(study_id)
    all_sessions.sort(key=lambda x: x.start_time)
    all_devices = e4.study_device_usage(study_id)
    # prepare plot
    f = plt.figure("Duration of Empatica records uploaded in hours per participant")
    ax = plt.subplot(111)
    # for each correspondances eg for each participant
    for c in correspondences:
        c_uuid = c.get("survey_user_uuid")   
        c_empatica_sn = c.get("empatica_sn")   
        c_start_date = datetime.strptime(c.get("start_date") + " 12:00:00", '%d/%m/%Y %H:%M:%S')
        c_end_date = datetime.strptime(c.get("end_date") + " 11:59:59", '%d/%m/%Y %H:%M:%S')
        # find the hardware_code corresponding to the empatica serial number
        c_hardware_code = None
        for d in all_devices:
            if c_empatica_sn == d.serial_number or not c_empatica_sn:
                c_hardware_code = d.hardware_code
                break
        # collected data
        x = []
        y = []
        # find survey_user's session, check session's device_id and session's date
        for s in all_sessions:
            if s.device_id == c_hardware_code and c_start_date <= datetime.fromtimestamp(s.start_time) <= c_end_date and s.duration >= 0:
                x.append(datetime.fromtimestamp(s.start_time))
                y.append(y[-1] if len(y) > 0 else 0)
                x.append(datetime.fromtimestamp(s.start_time + s.duration))
                y.append(y[-1] + (s.duration / 3600))
        # plot result
        if show_points:
            plt.plot(x, y, "o:", label="uuid " + c_uuid)
        else:
            plt.plot(x, y, label="uuid " + c_uuid)
    # adapt the legend to the number of uuids
    if len(correspondences) > 5:
        # Shrink current axis by 20% and put a legend to the right of the current axis
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else :
        plt.legend()
    # final touches
    plt.gcf().autofmt_xdate() # beautify the x-labels
    plt.xlabel("Time")
    plt.ylabel("Duration of Empatica records uploaded in hours")
    plt.show()
    # end
    return f


def number_of_survey_answers_per_survey_user_and_by_time(correspondences : list[dict]):
    """
    Plots the evolution of the volume of survey_answer posted by participant as a function of time.
    + correspondences : correspondence between survey_user_uuid, search code and Empatica serial number for the participants you want to analyze
    """
    # find the uuids of the survey_users that interest us
    survey_user_uuids = []
    for c in correspondences:
        survey_user_uuids.append(c.get("survey_user_uuid"))
    # prepare plot
    title = "titre ici"
    f = plt.figure(title)
    ax = plt.subplot(111)
    # for each participant
    for uuid in survey_user_uuids:
        # retrieve the participant's answers sorted by dates
        query = "SELECT * FROM survey_answer WHERE user_id = '" + uuid + "' ORDER BY answer_date"
        survey_user_s_survey_answers = get_survey_answer(request=query)
        # collected data
        x = []
        y = []
        answer_count = 0
        for a in survey_user_s_survey_answers:
            answer_count += 1
            x.append(a.answer_date)
            y.append(answer_count)
        plt.plot(x, y, label="uuid " + uuid)
    # adapt the legend to the number of uuids
    if len(correspondences) > 5:
        # Shrink current axis by 20% and put a legend to the right of the current axis
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else :
        plt.legend()
    # final touches
    plt.gcf().autofmt_xdate() # beautify the x-labels
    plt.xlabel("Time")
    plt.ylabel("Number of survey_answer")
    plt.show()
    # end
    return f 




## Check the overlapping of sessions in time ##


def check_overlapping_of_sessions_in_time(study_id) -> list[plt.figure]:
    """
    + study_id : empatica corresponding study ID
    """
    # retrieve all devices and session of the study
    e4 = MyE4Client()
    devices = e4.study_device_usage(study_id = study_id)
    sessions = e4.study_sessions_list(study_id = study_id)
    sessions.sort(key=lambda x: x.start_time)
    # collected data
    figures = list()
    # for each device
    for d in devices:
        # collected data
        device_overlaps = list()
        # retrieve all device's sessions
        device_sessions = list()
        for s in sessions:
            if s.device_id == d.hardware_code:
                device_sessions.append(s)
        # Check if the sessions associated with the device overlap in time, compare with all other sessions
        for s_index, s in enumerate(device_sessions):
            s_end_timestamp = s.start_time + s.duration
            for s_bis in device_sessions[s_index+1:]:
                s_bis_start_timestamp = s_bis.start_time
                s_bis_end_timestamp = s_bis_start_timestamp + s_bis.duration
                # if overlap
                if s_bis_start_timestamp < s_end_timestamp:
                    end_timestamp = None
                    if s_bis_end_timestamp > s_end_timestamp : # normal case
                        end_timestamp = s_end_timestamp
                    else :
                        end_timestamp = s_bis_end_timestamp
                    device_overlaps.append( (s, s_bis, abs(s_bis_start_timestamp - end_timestamp)) )
        # plot only if we have found overlaps
        if len(device_overlaps) > 0:
            title = "Sessions that overlap in time for the device " + str(d.serial_number)
            f = plt.figure(title)
            ax = plt.subplot(111)
            y_offset = 0
            padding = 0.4
            line_height = padding / 3.0
            for tuple in device_overlaps:
                y_offset += 1
                # plot s
                s = tuple[0]
                s_start = datetime.fromtimestamp(s.start_time)
                s_end = datetime.fromtimestamp(s.start_time + s.duration)
                x = [s_start, s_end]
                y = [y_offset, y_offset] 
                plt.fill_between(x, y_offset-(line_height/2.0), y_offset+(line_height/2.0), label = str(timedelta(seconds=s.duration)))
                plt.text(datetime.fromtimestamp((s.start_time * 2 + s.duration) / 2.0), y_offset, s.id, color="black", horizontalalignment="center", verticalalignment="center")
                # plot s_bis
                s_bis = tuple[1]
                s_bis_start = datetime.fromtimestamp(s_bis.start_time)
                s_bis_end = datetime.fromtimestamp(s_bis.start_time + s_bis.duration)
                x = [s_bis_start, s_bis_end]
                y = [y_offset + padding, y_offset + padding]
                plt.fill_between(x, y_offset + padding -(line_height/2.0), y_offset + padding + (line_height/2.0), label = str(timedelta(seconds=s_bis.duration)))
                plt.text(datetime.fromtimestamp((s_bis.start_time * 2 + s_bis.duration) / 2.0), y_offset + padding, s_bis.id, color="black", horizontalalignment="center", verticalalignment="center")
                #  plot overlap and its duration
                x = [s_bis_start, s_bis_start]
                y = [y_offset + (line_height/2.0), y_offset + padding- (line_height/2.0)]
                plt.plot(x, y, "--", color="red")
                end = None
                if s_bis_end > s_end : # normal case
                    end = s_end
                else :
                    end = s_bis_end       
                x = [end, end]
                y = [y_offset + (line_height/2.0), y_offset + padding- (line_height/2.0)]
                plt.plot(x, y, "--", color="red")
                x = datetime.fromtimestamp((s_bis_start.timestamp() + end.timestamp()) /2.0)
                y = (y_offset + y_offset + padding) / 2.0
                overlap_s_duration = str(timedelta(seconds=tuple[2]))
                plt.text( x, y, overlap_s_duration, horizontalalignment="center", verticalalignment="center")   
            plt.gcf().autofmt_xdate() # beautify the x-labels
            plt.xlabel("Time")
            ylim_bottom, ylim_top = plt.ylim() 
            plt.ylim(bottom = ylim_bottom - padding, top = ylim_top + padding)
            plt.legend()
            plt.legend(handletextpad=2)
            plt.title(title)
            plt.show()
            figures.append(f)
    return figures






if __name__ == '__main__':


    study_id = 3328 

    number_of_day = 100

    make_pdf_from_figures(
        figures=check_overlapping_of_sessions_in_time(study_id=study_id),
        destination_dir=downloads_directory_path,
        pdf_name="test"
    ) 