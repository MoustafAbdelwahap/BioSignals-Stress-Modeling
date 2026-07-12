import os
import pandas as pd


from db_access import *
from my_paths import *




def for_psycho_V2(destination_dir_path : str):
    """
    This method generates an excel file containing the responses of all participants to the presurvey. 
    These answers are originally written in full. 
    Then they are transcribed in the form of a score. 
    + destination_dir_path : excel's destination folder
    """
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(os.path.join(downloads_directory_path, "reponses_au_formulaire_preliminaire.xlsx"), engine='xlsxwriter')
    # retrieve all survey_answer and the presurvey
    survey_answers = get_survey_answer("SELECT * FROM survey_answer WHERE  user_id > '15' AND survey_id = '30' ORDER BY user_id ASC")
    # result dataframe
    result_df = pd.DataFrame()
    # concat all survey's answer in a dataframe
    for s_a in survey_answers:
        result_df = pd.concat([result_df, s_a.to_dataframe()], sort=False)
    # write in new excel sheet
    result_df.to_excel(writer, sheet_name='En_toute_lettre')
    # transpose writed answer to a score
    presurvey_s_questions = get_survey("SELECT * FROM survey WHERE uuid = '30'")[0].get_all_survey_s_questions()
    # for each question, i.e. for each column all groups combined
    for question_title in result_df.columns.get_level_values(1):
        # retrieve possible answers
        possible_answers = None 
        for q in presurvey_s_questions:
            if q.title == question_title and q.type == Question_Type.Choices :
                possible_answers = q.get_possible_answers()
                break
        if possible_answers :
            # change the value of each participant's answer for this question from a letter answer to a score (int)
            for user_uuid in result_df.index :
                letter_answer = result_df.loc[user_uuid, result_df.columns.get_level_values(1)==question_title]
                result_df.loc[user_uuid, result_df.columns.get_level_values(1)==question_title] = possible_answers.index(letter_answer[0])
    # write in new excel sheet
    result_df.to_excel(writer, sheet_name='Sous_forme_de_score')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

        
    


if __name__ == '__main__':

    for_psycho_V2(downloads_directory_path)



