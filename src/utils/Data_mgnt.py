import os
import yaml
import logging
import time
import pandas as pd
import json
import urllib.request as req
from zipfile import ZipFile
from src.utils.common import create_directories




def download_data(data_url,data_path):
    if not os.path.isfile(data_path):
        logging.info("downloading data...")
        filename,headers=req.urlretrieve(data_url,data_path)
        logging.info(f"filename: {filename} created with info \n{headers}")
    else:
        logging.info(f"file is already present")

def extract_data(unzip_data_folder_path,unzip_data_path):
    
    if not os.path.exists(unzip_data_folder_path):
        create_directories([unzip_data_folder_path])
        with ZipFile(unzip_data_path) as zip_:
            logging.info(f'data extraction started')
            zip_.extractall(unzip_data_folder_path)
    else:
        logging.info(f'Data already extracted')
