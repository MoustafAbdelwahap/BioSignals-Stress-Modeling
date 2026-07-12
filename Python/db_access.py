
import csv
from datetime import datetime
import re
import pg
from enum import Enum
import pandas as pd
import json
import os


db_connection_config_path = os.path.join("CSV", "db_connection_config.csv")



## Survey_user ##

class Survey_user:
    """ uuid, survey_code, data """
    def __init__(self, uuid, survey_code, data) -> None:
        self.uuid = uuid
        self.survey_code = survey_code
        self.data = data
    
    def __str__(self) -> str:
        return "user_" + str(self.uuid)




## Survey and additionnal classes ##

class Question_Type(Enum):
     Choices = "Choices"
     Range = "Range"
     Text = "Text"


class Question:
    """
    A question and its possible answers.
    Note : There are different types of questions (Choices, Range, Text)
    """
    def __init__(self, dictionnary : dict) -> None:
        self.answer = None
        self.dictionnary = dictionnary
        self.title = str(dictionnary.get("title"))
        self.type = Question_Type(dictionnary.get("type"))
        self.display = str(dictionnary.get("display"))

    def get_possible_answers(self) -> list[str] :
        if self.type == Question_Type.Choices :
            return self.dictionnary.get("data").get("answers")
        else :
            print("Error, the question isn't a choices question...")
            return None


class Question_Group:
    """
    Subpart of a Survey_content. This class groups one or more Question.
    """
    def __init__(self, dictionnary : dict) -> None:
        self.title = str(dictionnary.get("title"))
        self.question_list = list()
        for question_dict in dictionnary.get("data"):
            self.question_list.append(Question(dictionnary=question_dict))

    def get_questions(self) -> list[Question]:
        return self.question_list


class Survey_content:
    """
    Class that represents the content of the Survey, i.e. what the JSON file contains.
    It is the original survey passed by participants, without the metadata that goes around it.
    To summarize, a Survey is the most complete class, it contains additional information and a Survey_content.
    The Survey_content contains the information contained in the JSON file of the survey.
    It is composed of X Groups, which are themselves composed of Y Question.
    This last class contains a question itself.
    """
    def __init__(self, lst_of_dict : list[dict]) -> None:
        self.group_list = list()
        for group_as_dict in lst_of_dict:
            self.group_list.append(Question_Group(dictionnary=group_as_dict))

    def get_groups(self) -> list[Question_Group] :
        return self.group_list


class Survey:
    """ uuid, id, title, start_date, end_date, repeat_interval, form_definiton """
    def __init__(self, uuid, id, title, start_date, end_date, repeat_interval, form_definiton) -> None:
        self.uuid = uuid
        self.id = id
        self.title = title
        self.start_date = start_date
        self.end_date = end_date
        self.repeat_interval = repeat_interval
        self.form_definition = form_definiton
        """ list of dict """
        self.survey_content = Survey_content(lst_of_dict=form_definiton)

    def get_all_survey_s_questions(self) -> list[Question]:
        """
        Returns a list of all the questions of the survey
        """
        result = list()
        for g in self.survey_content.get_groups():
            for q in g.get_questions():
                result.append(q)
        return result

    def __str__(self) -> str:
        return "survey_" + str(self.uuid)




## Survey_answer and additionnal classes ##

class Question_Answer :
    """
    A question and its answer.
    + dictionnay : 
    """
    def __init__(self, dictionnary) -> None:
        self.question = str(dictionnary.get("title"))
        self.type = str(dictionnary.get("type"))
        self.answer = str(dictionnary.get("data"))

    def get_question_and_answer(self) -> list[str]:
        """
        Return the question and its answer in a list.
        """
        return [self.question, self.answer]

    def __str__(self) -> str:
        return "Q: " + self.question + " / A: " + self.answer


class Question_Answer_Group :
    """
    Subpart of a Survey_answer_content. This class groups one or more Question_answer.
    + dictionnary :
    """
    def __init__(self, dictionnary : dict) -> None:
        self.title = str(dictionnary.get("title"))
        self.question_answer_list = list()
        for question_answer in dictionnary.get("data"):
            self.question_answer_list.append(Question_Answer(question_answer))

    def get_question_answers(self) -> list[Question_Answer]:
        return self.question_answer_list

    def get_questions(self) -> list[str]:
        """
        Returns the list of questions from the group.
        """
        return [q_a.question for q_a in self.question_answer_list]

    def get_answers(self) -> list[str]:
        """
        Returns the list of answers from the group.
        """
        return [q_a.answer for q_a in self.get_question_answers()]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Example :
            
            Group's name
            Q1  Q2  Q3 ... Qn     
          0 A1  A2  A3 ... An
        """
        cols = pd.MultiIndex.from_product([[self.title], self.get_questions()])
        data = [self.get_answers()]
        df = pd.DataFrame(data, columns=cols)
        return df

    def __str__(self) -> str:
        return str(self.to_dataframe())
       

class Survey_answer_content :
    """
    Class that represents the content of the Survey_answer, i.e. what the JSON file contains.
    It is the answer given by the participant itself, without the metadata that goes around it.
    To summarize, a Survey_Answer is the most complete class, it contains additional information and a Survey_answer_content.
    The Survey_anwer_content contains the information contained in the JSON file of a response.
    It is composed of X Groups, which are themselves composed of Y Question_answer.
    This last class contains a question and its answer.
    + dictionnary : 
    """
    def __init__(self, dictionnary : dict) -> None:
        self.dictionnary = dictionnary
        self.survey_title = str(dictionnary.get("title"))
        self.group_list = list()
        for group_under_string_format in dictionnary.get("data"):
            self.group_list.append(Question_Answer_Group(json.loads(group_under_string_format)))

    def get_groups(self) -> list[Question_Answer_Group]:
        return self.group_list

    def get_groups_names(self) -> list[str]:
        """
        Returns the list of group names.
        """
        return [g.title for g in self.get_groups()]
    
    def to_dataframe(self) -> pd.DataFrame :
        """
        Example :
            
            Group1      Group2          Group_n
            Q1  Q2  Q3  Q4  Q5  Q6 ...  Qn     
          0 A1  A2  A3  A4  A5  A6 ...  An
        """
        result = pd.DataFrame()
        for g in self.get_groups():
            result = pd.concat([result, g.to_dataframe()], axis=1)
        return result


class Survey_answer:
    """ uuid, survey_id, user_id, attempt_nb, answer_date, answer """
    def __init__(self, uuid, survey_id, user_id, attempt_nb, answer_date, answer) -> None:
        self.uuid = uuid
        self.survey_id = survey_id
        self.user_id = user_id
        self.attempt_nb = attempt_nb
        self.answer_date = datetime.strptime(answer_date.split(".")[0], '%Y-%m-%d %H:%M:%S') 
        """ datetime """
        self.answer = Survey_answer_content(dictionnary=answer)

    def to_dataframe(self):
        """
        Example :
            
                    Group1      Group2          Group_n
        user_uuid   Q1  Q2  Q3  Q4  Q5  Q6 ...  Qn     
                0   A1  A2  A3  A4  A5  A6 ...  An
        """
        df = self.answer.to_dataframe()
        df.index = pd.Index([int(self.user_id)], name="user UUID")
        return df

    def __str__(self) -> str:
        return "answer_" + str(self.uuid)




## Pre-made requests  ##

class Queries(Enum):
    """
    Pre-made requests, example :
        users = get_survey_user(Queries.get_all_survey_user.value)
    """
    get_all_survey_user = "SELECT * FROM survey_user"
    get_all_survey = "SELECT * FROM survey"
    get_all_survey_answer = "SELECT * FROM survey_answer"




## Methods to connect to the database ##

def get_db_connection_config() -> dict:
    """ Loads the secret information needed to connect to the database. """
    connection_config = {}
    with open(db_connection_config_path, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            connection_config[row[0]] = row[1]
    return connection_config


def get_connection() -> pg.DB:
    """ Retrieves a connection to the database. Warning: do not forget to close it afterwards. """
    connection_config = get_db_connection_config()
    connection = pg.DB(
        host = connection_config.get("host"),
        user = connection_config.get("user"),
        passwd = connection_config.get("passwd"),
        dbname = connection_config.get("dbname"),
        port = int(connection_config.get("port"))
    )
    return connection




## Methods to make requests ##

def make_request(request : str):
    """
    Allows you to make a query to the database and retrieve the result.
    Notes: It can be useful to use the "getresult()" method on this method result.
    + request : any SQL request
    """
    connection = get_connection()
    result = connection.query(request)
    connection.close()
    return result


def get_survey_user(request : str) -> list[Survey_user]:
    """
    Returns objects of type Survey_user directly from a query.
    + request : SQL query that returns Survey_user, example : "SELECT * FROM survey_user"
    """
    result = make_request(request)
    users = []
    for user in result:
        users.append(Survey_user(user[0], user[1], user[2]))
    return users


def get_survey(request : str) -> list[Survey]:
    """
    Returns objects of type Survey directly from a query.
    + request : SQL query that returns Survey, example : "SELECT * FROM survey"
    """    
    result = make_request(request)
    surveys = []
    for survey in result:
        surveys.append(Survey(survey[0], survey[1], survey[2], survey[3], survey[4],  survey[5],  survey[6]))
    return surveys


def get_survey_answer(request : str) -> list[Survey_answer]:
    """
    Returns objects of type Survey_answer directly from a query.
    + request : SQL query that returns Survey_answer, example : "SELECT * FROM survey_answer"
    """    
    result = make_request(request)
    answers = []
    for answer in result:
        answers.append(Survey_answer(answer[0], answer[1], answer[2], answer[3], answer[4], answer[5]))
    return answers



## Other useful methods ##

def get_survey_user_uuid(research_code : str) -> str:
    """
    Returns the uuid of the survey_user who owns the research code passed in parameter (None if the research code is not found).
    + research_code : research_code
    """
    request = "SELECT uuid FROM survey_user WHERE survey_code = '"+ research_code +"'"
    user_id = re.sub("[^0-9]", "", str(make_request(request).getresult()))
    if user_id == "":
        return None
    else: 
        return user_id
