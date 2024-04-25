import numpy as np
import cv2
from PIL import Image
import pandas as pd
import os
from loadModel import model as model
import torch


model = model.to("cuda")


metaData=pd.read_csv(r"C:\newDataset\metaDataTrain.csv")
oldLabels=[]
newLabels=[]
for row in metaData.values:
    oldPath=os.path.join(r"C:\data\new_OPA\composite\train_set",row[5].split("/")[-1])
    print(row[5].split("/")[-1], row[-1])
    oldImg=Image.open(oldPath).resize((512,512)).convert("RGB")
    newPath=os.path.join(r"C:\newDataset\images",str(row[-1])+".jpg")
    newImg=Image.open(newPath).resize((512,512)).convert("RGB")
    with torch.no_grad():
        _,labelOld,__=model(oldImg)
        tup1=(_,labelOld,__)
        print(labelOld,__)
        _,labelNew,__=model(newImg)
        tup2=(_,labelNew,__)
        print(labelNew,__)
    oldLabels.append(labelOld)
    newLabels.append(labelNew)

oldLabels=np.array(oldLabels)
newLabels=np.array(newLabels)
print(sum(oldLabels==0))
print(sum(newLabels==0))