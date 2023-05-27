import os
import yaml
import logging
import time
import pandas as pd

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

def count_params(model):
    model_params={"Modules":list(),"parameters":list()}
    total={'trainable':0,'non-trainable':0}

    for name,parameters in model.named_parameters():
        params=parameters.numel()
        if not parameters.requires_grad:
            total['non-trainable']+=params
        else:
            model_params['Modules'].append(name)
            model_params['parameters'].append(params)
            total['trainable']+=params
    df = pd.DataFrame(model_params)
    
    return df,total

def training(model,train_data_loader,num_eopch,device,optimizer,criterion):
    model.to(device)
    for epoch in range(num_eopch):
        with tqdm(train_data_loader) as tqm_loader:
            for image,label in tqm_loader:
                tqm_loader.set_description(f"Epoch {epoch + 1}/{num_eopch}")
                image=image.to(device)
                label=label.to(device)

                pred=model(image)
                loss=criterion(pred,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tqm_loader.set_postfix(loss=loss.item())
    return loss.item()

def evaluate(test_data_loader,device,model):
    pred=np.array([])
    target=np.array([])
    with torch.no_grad():
        for image,label in test_data_loader:
            image=image.to(device)
            label=label.to(device)
            y_pred=F.softmax(model(image),1)
            pred=np.concatenate((pred,torch.argmax(y_pred,1).cpu().numpy()))
            target=np.concatenate((target,label.cpu().numpy()))
    return pred,target

def prediction_viz(img,std,mean,device,model,inv_label_map,label,FIG_SIZE=(10,15)):
    img = img.unsqueeze(0).to(device)
    pred_prob=F.softmax(model(img),dim=1)
    argmax = torch.argmax(pred_prob).item()
    logging.info(f'Actual Label:{inv_label_map[label.item()]},pred label:{inv_label_map[argmax]}')

    pred_prob = pred_prob.cpu().data.numpy().squeeze()
    
    _, (ax1, ax2) = plt.subplots(figsize=FIG_SIZE, ncols=2)
    
    # unnormalize the img
    img = img.cpu()*np.array(std)[:, None, None] + np.array(mean)[:, None, None]
    ax1.imshow(img.squeeze().permute(1,2,0).cpu())
    ax1.axis("off")
    
    ax2.barh(np.arange(2), pred_prob)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(2))
    
    ax2.set_yticklabels(inv_label_map.values(), size="small")
    ax2.set_title("pred prob")
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()






