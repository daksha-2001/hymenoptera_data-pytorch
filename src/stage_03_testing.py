import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random

import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
plt.style.use('fivethirtyeight')
from src.utils.model_creation import evaluate,prediction_viz


STAGE = "Model Testing" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    img_size=config['model']['img_size']
    mean=config['model']['mean']
    std=config['model']['std']
    batch_size=config['model']['batch_size']
    test_folder=config['model']['test_path_folder']
    data_dir=config['data']['local_dir']
    unzip_folder_name=config['data']['unzip_folder_name']
    unzip_folder_path=os.path.join(data_dir,unzip_folder_name)
    model_save=config['model']['model_save']
    test_path=os.path.join(unzip_folder_path,test_folder)
    device='cuda' if torch.cuda.is_available() else "cpu"

    test_transform=transforms.Compose([transforms.Resize(tuple(img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
    ])
    test_dataset=datasets.ImageFolder(test_path,test_transform)
    test_data_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    model_save_path=os.path.join(unzip_folder_path,"hymenoptera_data\models",model_save)

    model=torch.load(model_save_path)
    
    pred,target=evaluate(test_data_loader=test_data_loader,device=device,model=model)

    logging.info("Model evaluation done")
    cm=confusion_matrix(target,pred)

    cm_string = '\n'.join(['\t'.join([str(cell) for cell in row]) for row in cm])

    # Log the confusion matrix using logging.info()
    logging.info(f"Confusion Matrix:\n{cm_string}")

    
    label_map=test_dataset.class_to_idx

    inv_label_map={label_map[key]:key for key in label_map}

    plt.figure(figsize=(20,20))
    g=sns.heatmap(cm,annot=True,fmt='d')

    data=next(iter(test_data_loader))
    image,label=data    

    prediction_viz(img=image[2],inv_label_map=inv_label_map,std=std,mean=mean,device=device,model=model,
                   label=label[2])
    



    

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