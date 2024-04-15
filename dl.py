import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image
import json

class Coco(Dataset):
    def __init__(self, dataPath, annotationPath, transform=None):
        super(Coco, self).__init__()
        np.random.seed(0)
        self.transform = transform
        self.datas, self.num_classes = self.loadFromMemory(
            dataPath, annotationPath)
        self.dataPath = dataPath
        self.annotationPath = annotationPath

    def loadFromMemory(self, dataPath, annotationPath):
        datas = {}
        classes = 0
        # agrees = [0, 90, 180, 270]
        with open(annotationPath, 'r') as f:
            jsonData = json.load(f)
            classes = len(jsonData.keys())
            
            for id in jsonData.keys():
                datas[id] = []

            # for agree in agrees:
            for id in jsonData.keys():
                filenameAndBbox = jsonData[id]
                for filename in filenameAndBbox.keys():
                    image = Image.open(os.path.join(dataPath, filename))
                    bbox = filenameAndBbox[filename]
                    for box in bbox:
                        box = [int(x) for x in box]

                        left = box[0]
                        top = box[1]
                        right = box[0] + box[2]
                        bottom = box[1] + box[3]
                        img = image.crop((left, top, right, bottom)).convert("RGB")
                        w, h = img.size
                        if w == 0 or h == 0:
                            continue
                        datas[id].append(img)

        print("finish loading training dataset to memory")
        return datas, classes
    
    def __len__(self):
        count = 0
        for key in self.datas:
            count += len(self.datas[key])
        return count
    
    def __getitem__(self, index):
        label = None
        image1 = None
        image2 = None
        index = random.randint(1, 10)
        # Get image from the same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.choice(list(self.datas.keys()))
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])

        # Get images from different class
        else:
            label = 0.0
            ids = list(self.datas.keys())
            idx1 = random.choice(ids)
            idx2 = random.choice(ids)

            while idx1 == idx2:
                idx2 = random.choice(ids)
                
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])
        
        if self.transform:
            image1 = self.transform(image1).float()
            image2 = self.transform(image2).float()

        return image1, image2, torch.tensor(label).long()
    
