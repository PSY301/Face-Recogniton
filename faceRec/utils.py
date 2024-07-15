import os


def check_project(name_of_project: str) -> bool:
    """
    Checks if the name_of_project project exists.
    :param: name_of_project[str]
        name of project.
    :return:bool
        is the name_of_project exist?
    """
    return (os.path.exists(fr"{os.path.abspath(r'faceRec/dataSets')}\{name_of_project}") and
            os.path.exists(fr"{os.path.abspath(r'faceRec/trainers')}\{name_of_project}"))

