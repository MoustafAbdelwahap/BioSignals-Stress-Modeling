#https://support.empatica.com/hc/en-us/articles/201608896-Data-export-and-formatting-from-E4-connect-
# This notebook is used to acess the collected data through E4 connect (Empatica's Web Portal) which allows to allows researchers to visualize E4 session data, download sessions in CSV format
# We do not need it since we alerdy have all the data downloaded in our local device

import csv
from datetime import datetime
import json
import os
import re
from xmlrpc.client import DateTime
import requests
import shutil
import fnmatch
import tempfile



empatica_connection_config_path = os.path.join("CSV", "empatica_connection_config.csv")



## Classes to access Empatica's E4 data from E4 Connect plateform ##

class Empatica_Session:
    """
    Example :
            id : 1463639
            device_id : 9a2414
            duration : 20
            status : 0
            start_time : 1646428328
            label : 16810
            device : E4 3.3
            exit_code : 0
    """
    def __init__(self, session: dict) -> None:
        self.id = int(session.get("id"))
        self.device_id = str(session.get("device_id"))
        """ corresponding to the Device.hardware_code (I know it's strange but that's how Empatica works) """
        self.duration = int(session.get("duration"))
        """ in seconds """
        self.status = int(session.get("status"))
        self.start_time = int(session.get("start_time"))
        """ Unix timestamp """
        self.label = int(session.get("label")) if session.get("label") else None
        self.device = str(session.get("device"))
        """ device model + version number : E4 3.3 """
        self.exit_code = int(session.get("exit_code"))

    def __str__(self) -> str:
        return "session_" + str(self.id)

    def get_description(self) -> str:
        description = "Session\n\tid : " + str(self.id) + "\n"
        description += "\tdevice_id : " + self.device_id + "\n"
        description += "\tduration : " + str(self.duration) + "\n"
        description += "\tstatus : " + str(self.status) + "\n"
        description += "\tstart_time : " + str(self.start_time) + "\n"
        description += "\tlabel : " + str(self.label) + "\n"
        description += "\tdevice : " + self.device + "\n"
        description += "\texit_code : " + str(self.exit_code) + "\n"
        return description        


class Empatica_Study:
    def __init__(self, study: dict) -> None:
        self.id = study.get("id")
        self.name = study.get("name")
        self.description = study.get("description")
        self.uploader_name = study.get("uploader_name")
        self.uploader_password = study.get("uploader_password")
        self.created_at = study.get("created_at")
        self.is_closed = study.get("is_closed")
        self.zip_name = study.get("zip_name")
        self.tot_duration_per_day = study.get("tot_duration_per_day")
        self.days = study.get("days")
        self.avg_duration_per_day = study.get("avg_duration_per_day")
        self.devices = study.get("devices")
        self.avg_duration_per_device = study.get("avg_duration_per_device")
        self.new_sessions_count = study.get("new_sessions_count")
    
    def __str__(self) -> str:
        return "study_" + str(self.id)

    def get_description(self) -> str:
        description = "Study\n\tid : " + str(self.id) + "\n"
        description += "\tname : " + str(self.name) + "\n"
        description += "\tdescription : " + str(self.description) + " s\n"
        description += "\tuploader_name : " + str(self.uploader_name) + "\n"
        description += "\tuploader_password : " + str(self.uploader_password) + "\n"
        description += "\tcreated_at : " + str(self.created_at) + "\n"
        description += "\tis_closed : " + str(self.is_closed) + "\n"
        description += "\tzip_name : " + str(self.zip_name) + "\n"
        description += "\ttot_duration_per_day : " + str(self.tot_duration_per_day) + "\n"
        description += "\tdays : " + str(self.days) + "\n"
        description += "\tavg_duration_per_day : " + str(self.avg_duration_per_day) + "\n"
        description += "\tdevices : " + str(self.avg_duration_per_device) + "\n"
        description += "\tavg_duration_per_device : " + str(self.avg_duration_per_device) + "\n"
        description += "\tnew_sessions_count : " + str(self.new_sessions_count) + "\n"
        return description   


class Empatica_Device:
    def __init__(self, device : dict) -> None:
        self.device_id = device.get("device_id")
        self.hardware_code = device.get("hardware_code")
        self.device_label = device.get("device_label")
        self.device_model = device.get("device_model")
        self.sessions = device.get("sessions")
        self.tot_duration = int(device.get("tot_duration")) # in seconds
        self.serial_number = self.device_label_to_SN(self.device_label)

    def __str__(self) -> str:
        return "device_" + str(self.device_id)

    def device_label_to_SN(self, device_label) -> str:
        if(device_label):
            # int to hex string
            sn = hex(int(device_label))[2:]
            # leading with 0 and keep only the last 5 char         
            pad = "00000"
            sn = (pad + sn)[-5:]
            # add a "A" and switch to upper case                   
            sn = "A" + sn.upper()
            return sn
        else: 
            return None
            
    def get_description(self) -> str:
        description = "Device\n\tid : " + str(self.device_id) + "\n"
        description += "\thardware_code: " + str(self.hardware_code) + "\n"
        description += "\tdevice_label : " + str(self.device_label) + "\n"
        description += "\tdevice_model : " + str(self.device_model) + "\n"
        description += "\tsessions : " + str(self.sessions) + "\n"
        description += "\ttot_duration : " + str(self.tot_duration) + "\n"
        description += "\tserial_number : " + str(self.serial_number) + "\n"
        return description   





## How to get the necessary information to connect to the E4 Connect platform ##

def load_e4_connection_config() -> dict:
    """ Retrieves the username and password to connect to the E4 Connect platform. """
    connection_config = {}
    with open(empatica_connection_config_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            connection_config[row[0]] = row[1]
    return connection_config




## E4 Client ##

class MyE4Client:
    """
    Class that allows to interact with the E4 Connect platform. It allows to retrieve lists of sessions/studies and devices and
    to download sessions/studies data.
    """


    ## Getting Started ##

    def __init__(self, user: str = None, pwd: str = None):
        """
        If no username or password is passed, it uses the data from the "empatica_connection_config.csv" file.
        + user : username
        + pwd : password
        """
        self.s = requests.Session()
        self.user_id = None

        if user and pwd:
            self.auth(user, pwd)
        else: 
            config = load_e4_connection_config()
            self.auth(config.get("user"), config.get("pwd"))


    def auth(self, user: str, pwd: str):
        """
        Allows you to authenticate to the Empatica E4 Connect platform and retrieve the user's ID.
        Cookies are kept for future requests.
        + user : username
        + pwd : password
        """
        url = 'https://www.empatica.com/connect/authenticate.php'
        auth_info = {'username': user, 'password': pwd}
        self.s.post(url, auth_info).raise_for_status()

        url_sessions_main = 'https://www.empatica.com/connect/sessions.php'
        resp = self.s.get(url_sessions_main)
        resp.raise_for_status()
        self.user_id = re.search(r'userId = ([0-9]*);', resp.text).group(1)

        print("[E4 Client] Authentication successful, E4 client ready to use")
        

    def get_user_id(self) -> str:
        """ Returns the user ID. """
        return self.user_id




    ## Internal methods ##

    def __resp_to_sessions(self, resp_text : requests.Response.text) -> list[Empatica_Session]:
        """
        Transforms the response of a request into a list of Session objects. 
        + resp_text : requests.Response.text
        """
        sessions = []
        for item in json.loads(str(resp_text)):
            sessions.append(Empatica_Session(item))
        return sessions

    
    def __resp_to_studies(self, resp_text : requests.Response.text) -> list[Empatica_Study]:
        """
        Transforms the response of a request into a list of Study objects. 
        + resp_text : requests.Response.text
        """
        studies = []
        for item in json.loads(str(resp_text)):
            studies.append(Empatica_Study(item))
        return studies

    
    def __resp_to_devices(self, resp_text : requests.Response.text) -> list[Empatica_Device]:
        """
        Transforms the response of a request into a list of Device objects. 
        + resp_text : requests.Response.text
        """
        devices = []
        for item in json.loads(str(resp_text)):
            devices.append(Empatica_Device(item))
        return devices


    ## User Sessions ##

    def user_sessions_list(self) -> list[Empatica_Session]:
        """ List of user sessions. """
        url = 'https://www.empatica.com/connect/connect.php/users/{uid}/sessions?from=0&to=999999999999'
        
        resp = self.s.get(url.format(uid=self.user_id))
        resp.raise_for_status()
        return self.__resp_to_sessions(resp.text)


    def download_user_session(self, session_id, file_path : str = '.'):
        """
        Downloads a session as a zip.
        Contents: ACC, BVP, EDA, HR, IBI, TEMP and two additional information files info.txt and tags.txt.
        + session_id : Session ID as int or String
        + file_path : destination
        """
        url = "https://www.empatica.com/connect/download.php?id={id}"
        file_path = str(file_path)
        session_id = str(session_id)

        resp = self.s.get(url.format(id=session_id))
        file_path = os.path.join(file_path, '%s.zip' % session_id) if os.path.isdir(file_path) else file_path
        with open('%s' % file_path, 'wb') as f:
            f.write(resp.content)



    ## Studies ##

    def studies_list(self) -> list[Empatica_Study]:
        """ List of studies. """
        url = "https://www.empatica.com/connect/connect.php/studies"
        
        resp = self.s.get(url)
        resp.raise_for_status()
        return self.__resp_to_studies(resp.text)


    def study_sessions_list(self, study_id) -> list[Empatica_Session]:
        """
        List of study sessions.
        + study_id : Study ID as int or String
        """
        url = "https://www.empatica.com/connect/connect.php/studies/{study_id}/sessions?from=0&to=999999999999999999999999999999"
        study_id = str(study_id)

        resp = self.s.get(url.format(study_id=study_id))
        resp.raise_for_status()
        return self.__resp_to_sessions(resp.text)


    def study_sessions_list_from_to_Dates(self, study_id, start_date : datetime , end_date : datetime) -> list[Empatica_Session]:
        """
        List of study sessions (date to date).
        + study_id : Study ID as int or String
        + start_date : start date, exemple = datetime.strptime("10/02/22 00:00:00", '%d/%m/%y %H:%M:%S')
        + end_date : end date, exemple = datetime.strptime("10/02/22 00:00:00", '%d/%m/%y %H:%M:%S')
        Note: The dates are in the form of 'timestamp' in the url of the request.
        """
        study_id = str(study_id)
        start_timestamp = str(datetime.timestamp(start_date))
        end_timestamp = str(datetime.timestamp(end_date))
        url = "https://www.empatica.com/connect/connect.php/studies/{study_id}/sessions?from={start_timestamp}&to={end_timestamp}"
        
        resp = self.s.get(url.format(study_id=study_id, start_timestamp=start_timestamp, end_timestamp=end_timestamp))
        resp.raise_for_status()
        return self.__resp_to_sessions(resp.text)


    def download_study_session_ZIP(self, session_id, path_destination_dir : str = ".", zip_name: str = None):
        """
        Downloads a session of a study as a zip.
        + session_id : Session ID as int or String
        + path_destination_dir : destination directory for the zip file
        + zip_name : name of the final zip file (without ".zip" at the end)
        """
        url = "https://www.empatica.com/connect/download.php?id={session_id}&fromStudy=true"
        session_id = str(session_id)
        path_destination_dir = str(path_destination_dir)
        if not zip_name :
            zip_name = "session_" + str(session_id)
        else :
            zip_name = str(zip_name)

        resp = self.s.get(url.format(session_id=session_id))
        file_path = os.path.join(path_destination_dir, '%s.zip' % zip_name)
        with open('%s' % file_path, 'wb') as f:
            f.write(resp.content)
        
        print("[E4 Client]", file_path, "downloaded")

    
    def download_all_study_sessions_ZIP(self, study_id, path_destination_dir : str = ".", zip_name: str = None):
        """
        Downloads all sessions of a study as a zip.
        + study_id : Study ID as int or String 
        + path_destination_dir : destination directory for the zip file 
        + zip_name : name of the final zip file (without ".zip" at the end)
        """
        study_id = str(study_id)
        path_destination_dir = str(path_destination_dir)
        if not zip_name:
            zip_name = "study_" + str(study_id)
        else:
            zip_name = str(zip_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            study_dir = os.path.join(temp_dir, zip_name)
            os.mkdir(study_dir)

            for session in self.study_sessions_list(study_id):
                self.download_study_session_ZIP(session.id, study_dir, str(session))

            files = os.listdir(study_dir)
            pattern = "*.zip"
            zip_files = fnmatch.filter(files, pattern)
            for zip in zip_files :
                zip_path = os.path.join(study_dir, zip)
                extracted_path = zip_path[:len(zip_path) - 4]
                shutil.unpack_archive(zip_path, extracted_path)
                os.remove(zip_path)

            result = shutil.make_archive(
                base_name=os.path.join(path_destination_dir, zip_name),
                format='zip',
                root_dir=study_dir,
            )

            print("[E4 Client]", result, "downloaded")



    ## Devices ##
                
    def study_device_usage(self, study_id):
        """
        List of study devices and their details.
        + study_id : Study ID as int or String 
        """
        url = "https://www.empatica.com/connect/connect.php/studies/{study_id}/stats?from=0&to=99999999999999"
        study_id = str(study_id)
        
        resp = self.s.get(url.format(study_id=study_id))
        resp.raise_for_status()
        return self.__resp_to_devices(resp.text)

                        
    def study_device_usage_from_to_Dates(self, study_id, start_date : DateTime, end_date : DateTime):
        """
        List of study devices and their details from one date to another.
        + study_id : Study ID as int or String
        + start_date : start date, exemple = datetime.strptime("10/02/22 00:00:00", '%d/%m/%y %H:%M:%S')
        + end_date : end date, exemple = datetime.strptime("10/02/22 00:00:00", '%d/%m/%y %H:%M:%S')
        Note: The dates are in the form of 'timestamp' in the url of the request.
        """
        url = "https://www.empatica.com/connect/connect.php/studies/{study_id}/stats?from={start_timestamp}&to={end_timestamp}"
        study_id = str(study_id)
        start_timestamp = str(datetime.timestamp(start_date))
        end_timestamp = str(datetime.timestamp(end_date))

        resp = self.s.get(url.format(study_id=study_id, start_timestamp=start_timestamp, end_timestamp=end_timestamp))
        resp.raise_for_status()
        return self.__resp_to_devices(resp.text)


