#This notebook is use to download the data and creat the architecture[sessions, answers, weather]
# We do not need it since we alerdy have all the data downloaded in our local device

from e4_access import *   # it is another py file that enable us to connect to E4 connect
import my_paths



# * Step 1 :
def create_dataset_architecture(destination_directory : str, dataset_name : str) -> str:
    """
    dataset_folder [sessions, answers, weather]
    + destination_directory : directory where the dataset will be created
    + dataset_name : name we want to give to the dataset
    -> return dataset_directory_path
    """
    # dataset_folder
    dataset_directory_path = os.path.join(destination_directory, dataset_name)
    os.makedirs(dataset_directory_path)
    # answers_directory_name
    answers_directorie_path = os.path.join(dataset_directory_path, my_paths.answers_directory_name)
    os.makedirs(answers_directorie_path)
    # sessions_directory_name
    sessions_directorie_path = os.path.join(dataset_directory_path, my_paths.sessions_directory_name)
    os.makedirs(sessions_directorie_path)
    # weather_directory_name
    weather_directorie_path = os.path.join(dataset_directory_path, my_paths.weather_directory_name)
    os.makedirs(weather_directorie_path)
    # end 
    return dataset_directory_path


# * Step 2 :
def download_empatica_sessions_for_each_participant(empatica_study_id, correspondence_sn_code_path : str, sessions_directorie_path : str, session_minimum_duration_in_seconds : int) -> str:
    """
    This method download the sessions of each participant from the E4 Connect platform.
    - dataset_folder
        - answers_directory_name
            - file_X
            - file_X
        - sessions_directory_name
            - user_0
                - session_X
                - session_X
            - user_1
                - session_X
                - session_X

    + empatica_study_id : Study ID as int or String
    + correspondence_sn_code_path : CSV file containing the correspondences between research code and serial number of Empatica devices
    + sessions_directorie_path : path to the destination directory, the dataset directory that contains the sessions of each participant
    + session_minimum_duration_in_seconds : minimum duration required for a session to be downloaded
    
    >>> Example :
    download_empatica_sessions_for_each_participant(
        empatica_study_id=miams_study_id,
        correspondence_sn_code_path=correspondences_sn_code_path,
        sessions_directorie_path= my_paths.get_dataset_sessions_directory_path(dataset_path=dataset_directory_path),
        session_minimum_duration_in_seconds=60
    )
    """
    # variables for report
    empatica_sessions_downloaded = 0
    empatica_sessions_already_downloaded = 0
    empatica_serial_number_not_recognized = []

    # retrieve the list of all study sessions and devices from empatica E4 Connect
    e4 = MyE4Client()
    study_sessions_list = e4.study_sessions_list(empatica_study_id)
    study_devices_list = e4.study_device_usage(empatica_study_id)

    # load correspondences from the CSV file (list of dict)
    # CSV organisation : ID Session;uuid;Code;Empatica SN;HR Code;Date Debut;Date fin;SN Occurrence;SN Not Found;;Available;1;;Commentaire
    correspondences = []
    with open(correspondence_sn_code_path, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        spamreader.__next__()
        for row in spamreader :
            if(row[1] == ""):
                break
            else :
                correspondences.append({
                    "user_uuid" : int(row[1]),
                    "empatica_sn" : str(row[3]),
                    "participation_start_date" : datetime.strptime(row[5] + " 12:00:00", '%d/%m/%Y %H:%M:%S'),   # at noon because the launch meetings where they were given their bracelet were at noon
                    "participation_end_date" : datetime.strptime(row[6] + " 11:59:59", '%d/%m/%Y %H:%M:%S'),   # at noon because the end meetings where they returned their bracelets were at noon
                    "comment" : row[-1],
                })

    # each correspondence corresponds to a new participant 
    for c in correspondences:

        # retrieve the data we are interested in for this match
        user_uuid = c.get("user_uuid")                                                                     
        participation_start_date = c.get("participation_start_date")
        participation_end_date = c.get("participation_end_date")

        # create a directory by participant
        participant_directory_path = os.path.join(sessions_directorie_path, "user_" + str(user_uuid))
        if not os.path.exists(participant_directory_path):
            os.makedirs(participant_directory_path)

        ## Major point : Download Empatica sessions ##

        # retrieve the device's hardware code corresponding to it serial number
        empatica_sn = c.get("empatica_sn")
        hardware_code = None
        for device in study_devices_list:
            if device.serial_number == empatica_sn:
                hardware_code = device.hardware_code
                break

        # download the participants' sessions (if they are not already downloaded), eg:
        #   - check if we have found the hardware_code corresponding to the participant's wristband
        #   if yes, for each study's session :
        #       - check if the session is long enough
        #       - check if the session's device_id matches with this hardware_code
        #       - check if the session took place in the dates that correspond to those of the participant
        #   if yes :
        #       download the session in the right folder
        if hardware_code :
            for session in study_sessions_list:
                if session.duration >= session_minimum_duration_in_seconds :
                    session_date = datetime.fromtimestamp(int(session.start_time))
                    if session.device_id == hardware_code and participation_start_date <= session_date <= participation_end_date:
                        # Example : session_XXXX
                        session_name = str(session) 
                        # Example : '/Users/imvia/Downloads/MIAMS_Raw_Dataset/sessions/user_Y/session_XXXXXX'
                        file_path = os.path.join(participant_directory_path, session_name)
                        if not os.path.exists(file_path):
                            e4.download_study_session_ZIP(session.id, participant_directory_path, session_name)
                            zip_file = file_path + ".zip"
                            shutil.unpack_archive(zip_file, file_path)
                            os.remove(zip_file)
                            empatica_sessions_downloaded += 1
                        else:
                            empatica_sessions_already_downloaded += 1
        else:
            # if the corresponding hardware_code could not be found it may be because the wristband does not appear in the list of Empatica devices
            # the error will be reported in the report
            empatica_serial_number_not_recognized.append((user_uuid, empatica_sn))
    
    
    # report creation
    report = "[download_empatica_sessions_for_each_participant]\n"
    report += "\t Empatica serial number not recognized :\n"
    for index_code in empatica_serial_number_not_recognized:
        com = None
        for c in correspondences:
            if index_code[1] == c.get("empatica_sn"):
                com = c.get("comment")
                break
        report += "\t\t user " + str(index_code[0]) + ", SN Empatica " + str(index_code[1]) + ", commentaire : " + com + "\n"
    if len(empatica_serial_number_not_recognized) == 0 :
        report += "\t\t no one\n"
    report += "\t Empatica sessions :\n"
    report += "\t\t already downloaded --> " + str(empatica_sessions_already_downloaded) + "\n"
    report += "\t\t downloaded --> " + str(empatica_sessions_downloaded) + "\n"
    return report


# * Step 3 :
# retrieve survey_answers --> thierry



# * Step 4 :
# Download weather data from "Météo France" 




def raw_dataset_generation():

    # MIAMS Study'ID on E4 Connect
    miams_study_id = 3328

    # Path of the file containing the correspondences between Empatica serial number and research code
    # for each participant of the experiment 
    correspondences_sn_code_path = os.path.join("CSV", "correspondance_sn_codes.csv")

    raw_dataset_name = "MIAMS_Raw_Dataset"

    dataset_directory_path = create_dataset_architecture(
        destination_directory=my_paths.downloads_directory_path,
        dataset_name=raw_dataset_name
    )

    download_empatica_sessions_for_each_participant(
        empatica_study_id=miams_study_id,
        correspondence_sn_code_path=correspondences_sn_code_path,
        sessions_directorie_path= my_paths.get_dataset_sessions_directory_path(dataset_path=dataset_directory_path),
        session_minimum_duration_in_seconds=60
    )
