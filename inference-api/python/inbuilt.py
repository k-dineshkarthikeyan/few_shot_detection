import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
from fastapi import FastAPI, UploadFile, File

class Network(nn.Module):
    def __init__(self, model, backbone='resnet18', path='../intern_dataset/Train/cropped_product_images') -> None:
        super(Network, self).__init__()

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize([224,224], antialias=True)
        ])
        weights_and_biases = model.state_dict()

        if backbone not in models.__dict__:
            raise Exception(
                f"No model named {backbone} exists in torchvision.models")

        # self.backbone = models.__dict__[backbone](weights=ResNet50_Weights.DEFAULT)

        self.backbone = resnet18(weights=None)
        self.backbone.eval()
        self.out = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),

            nn.Linear(512, 64),
            nn.Sigmoid(),

            nn.Linear(64, 32),
            nn.Sigmoid(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        backbone_wb = {k.replace('backbone.', ''): v for k,
                       v in weights_and_biases.items() if 'backbone' in k}
        out_wb = {k.replace('out.', ''): v for k,
                  v in weights_and_biases.items() if 'out' in k}
        self.backbone.load_state_dict(backbone_wb)
        self.out.load_state_dict(out_wb)
        self.path = path
        self.images = self.load_images(self.path)
        self.encodings = self.find_outs(self.images)

    def load_images(self, path):
        images = {}
        cat_names = os.listdir(path)
        for i in cat_names:
            images[i] = []
        for name in cat_names:
            for img in os.listdir(os.path.join(path, name)):
                image = Image.open(os.path.join(
                    path, name, img)).convert('RGB')
                if self.transforms:
                    image = self.transforms(image)
                images[name].append(image)
        return images

    def find_outs(self, images):
        encodings = {}
        for k in images:
            encodings[k] = []
        for k in images:
            for image in images[k]:
                image = image.unsqueeze(0)
                out = self.backbone(image)
                encodings[k].append(out)
        return encodings

    def forward(self, image):
        image = self.transforms(image)
        image = image.unsqueeze(0)
        out = self.backbone(image)
        print(out)
        probabs = {}

        for k in self.encodings:
            probabs[k] = []

        for k in self.encodings:
            for encoding in self.encodings[k]:
                dis = cosine_similarity(encoding, out).view(1, 1)
                output = self.out(dis)
                # output = torch.mean(output)
                probabs[k].append(output.item())
        # print(probabs)
        values = {}
        for k in probabs:
            values[k] = (max(probabs[k]))

        max_value = max(values.values())
        print(max_value)
        cat_name = ""
        for i in values:
            if values[i] == max_value:
                cat_name = i
                break

        return cat_name


class SiameseNetwork(nn.Module):
    def __init__(self, batch_size=1, backbone='resnet18'):
        super(SiameseNetwork, self).__init__()

        self.batch_size = batch_size

        if backbone not in models.__dict__:
            raise Exception(
                f"No model named {backbone} exists in torchvision.models")

        self.backbone = models.__dict__[backbone](
            weights=ResNet18_Weights.DEFAULT, progress=True)
        self.out = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),

            nn.Linear(512, 64),
            nn.Sigmoid(),

            nn.Linear(64, 32),
            nn.Sigmoid(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        feature1 = self.backbone(img1)
        feature2 = self.backbone(img2)
        output = cosine_similarity(feature1, feature2)
        output = self.out(output)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = FastAPI()
pretrained = torch.load('./model1.pt', map_location=device)
siamese = SiameseNetwork()
siamese.load_state_dict(pretrained)
path = r'D:\coding\internship\proglint\work\clean_json\intern_dataset\intern_dataset\Train\cropped_product_images'
model = Network(siamese, path=path)
model.eval()

@app.post('/inference')
def inference(images: list[UploadFile] = File(...)):
    l = []
    for image in images:
        img = Image.open(image.file).convert("RGB")
        out = model.forward(img)
        l.append(out)
    return {'result': l}
