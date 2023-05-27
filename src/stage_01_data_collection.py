import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.Data_mgnt import download_data,extract_data
import random
import os


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import urllib.request as req
from pathlib import Path
plt.style.use('fivethirtyeight')


STAGE = "Data Collection" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    data_dir=config['data']['local_dir']
    data_url=config['data']['data_url']
    data_zip_folder=config['data']['data_zip_folder']
    unzip_folder_name=config['data']['unzip_folder_name']

    create_directories([data_dir])

    data_path=os.path.join(data_dir,data_zip_folder)

    download_data(data_url,data_path)

    unzip_folder_path=os.path.join(data_dir,unzip_folder_name)
    
    extract_data(unzip_folder_path,data_path)





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e