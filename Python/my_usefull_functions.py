import os
import shutil

import my_paths



## To read in directories ##

def get_parquet_files_paths_from_a_directory(directory_path : str) -> list[str]:
    """
    Retrieves paths of each parquet files in a folder.
    + directory_path : directory's path
    """
    lst = list()
    for item in os.listdir(directory_path):
        if item.endswith(".parquet.gzip") :
            lst.append(os.path.join(directory_path, item))
    return lst




## To manage those damn files ##

def auto_extract_sessions_zip_files(directory_path : str, delete_zip_after_extract : bool):
    # check parameters
    if not os.path.exists(directory_path):
        print("Error,", directory_path, "doen't exist...")
        return
    # retrieves all directories and files
    path_list = my_paths.my_listdir(
        directory_path=directory_path,
        ignore_hidden_files=True,
        sort_by_creation_time=False
    )
    # for each of them
    for p in path_list:
        if p.endswith(".zip") :
            print(p)
            exctracted_path = p[:-4]
            shutil.unpack_archive(p, exctracted_path, "zip")
            if delete_zip_after_extract:
                os.remove(p)
        elif os.path.isdir(p) :
            auto_extract_sessions_zip_files(directory_path=p, delete_zip_after_extract=delete_zip_after_extract)

def files_extermination(directory_path : str, extension : str):
    """
    This method delete all ".zip" files in a directory (including subdirectories)
    + directory_path : folder to review
    + extension : extension of the files you want to delete. Example: ".zip" or ".gzip".
    """
    # check parameters
    if not os.path.exists(directory_path):
        print("Error,", directory_path, "doen't exist...")
        return
    # retrieves all directories and files
    path_list = my_paths.my_listdir(
        directory_path=directory_path,
        ignore_hidden_files=False,
        sort_by_creation_time=False
    )
    # for each of them
    for p in path_list:
        if p.endswith(extension) :
            os.remove(p)
        elif os.path.isdir(p) :
            files_extermination(directory_path=p, extension=extension)
