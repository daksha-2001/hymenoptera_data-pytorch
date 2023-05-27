import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.model_creation import count_params,training
import random

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


STAGE = "Training"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    train_folder=config['model']['train_path_folder']
    img_size=config['model']['img_size']
    num_epoch=config['model']['num_epoch']
    batch_size=config['model']['batch_size']
    model_save=config['model']['model_save']
    test_folder=config['model']['test_path_folder']
    data_dir=config['data']['local_dir']
    unzip_folder_name=config['data']['unzip_folder_name']

    mean=torch.tensor(config['model']['mean'])
    std=torch.tensor(config['model']['std'])

    

    device='cuda' if torch.cuda.is_available() else "cpu"

    unzip_folder_path=os.path.join(data_dir,unzip_folder_name)
    train_path=os.path.join(unzip_folder_path,train_folder)
    test_path=os.path.join(unzip_folder_path,test_folder)

    train_transform=transforms.Compose([transforms.Resize(tuple(img_size)),
                                        transforms.RandomRotation(degrees=20),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)
    ])
   
    

    train_dataset=datasets.ImageFolder(train_path,train_transform)
    logging.info("Train images Transformed")
    test_dataset=datasets.ImageFolder(test_path,test_transform)
    logging.info("Test images Transformed")

    label_map=train_dataset.class_to_idx
    logging.info(f'Label_map is {label_map}')

    train_data_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_data_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    train_data=next(iter(train_data_loader))

    model=models.alexnet(pretrained=True)
    logging.info(model)

    for param in model.parameters():
        param.requires_grad=False

    df,total=count_params(model)
    logging.info(f'Parameters count before modifying Classifier Layer: {total}\n {df.to_string()}')

    model.classifier=nn.Sequential(nn.Linear(in_features=9216,out_features=100,bias=True),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features=100, out_features=2, bias=True))
    df,total=count_params(model)
    logging.info(f'Parameters count after modifying Classifier Layer: {total}\n {df.to_string()}')

    model.to(device)

    logging.info(f'Model Device = {device} set')

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters())

    loss=training(model=model,train_data_loader=train_data_loader,num_eopch=num_epoch,device=device,criterion=criterion,optimizer=optimizer)

    logging.info(f'Training complete for {num_epoch} and loss is {loss}')

    create_directories(["hymenoptera_data\data\hymenoptera_data\models",])

    model_save_path=os.path.join(unzip_folder_path,"hymenoptera_data\models",model_save)
    
    torch.save(model,model_save_path)

    logging.info(f'Model saved at {model_save_path}')

   

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