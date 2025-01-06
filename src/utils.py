import sys
import logging
import os
import dill

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error Occurred in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno,str(error)   
    )
    return error_message 

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)  
        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message  
    
def save_object(file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)

            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'wb') as file_obj:
                dill.dump(obj, file_obj)

        except Exception as e:
            raise CustomException(e, sys)            