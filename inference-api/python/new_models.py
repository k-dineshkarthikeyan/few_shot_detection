from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torch.nn.functional import cosine_similarity
class SiameseNetwork(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(SiameseNetwork, self).__init__()

        if backbone not in models.__dict__:
            raise Exception(
                f"No model named {backbone} exists in torchvision.models")

        self.backbone = models.__dict__[backbone](
            weights=ResNet18_Weights.DEFAULT)
        self.backbone.eval()
        self.out = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),

            nn.Linear(512, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

class Network(nn.Module):
    def __init__(self, model, embeddings, backbone='resnet18') -> None:
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

        self.backbone = models.__dict__[backbone](
            weights=ResNet18_Weights.DEFAULT)
        self.backbone.eval()
        self.out = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),

            nn.Linear(512, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        backbone_wb = {k.replace('backbone.', ''): v for k,
                       v in weights_and_biases.items() if 'backbone' in k}
        out_wb = {k.replace('out.', ''): v for k,
                  v in weights_and_biases.items() if 'out' in k}
        self.backbone.load_state_dict(backbone_wb)
        self.out.load_state_dict(out_wb)
        self.encodings = embeddings

    def forward(self, image):
        image = self.transforms(image)
        image = image.unsqueeze(0)
        out = self.backbone(image)
        probabs = {}

        for k in self.encodings:
            probabs[k] = []

        for k in self.encodings:
            for encoding in self.encodings[k]:
                dis = cosine_similarity(encoding, out).view(1, 1)
                output = self.out(dis)
                probabs[k].append(output.item())
        values = {}
        for k in probabs:
            values[k] = (max(probabs[k]))

        max_value = max(values.values())
        print(max_value)
        if max_value < 0.90:
            return 'nil'
        cat_name = ""
        for i in values.keys():
            if values[i] == max_value:
                cat_name = i
                break

        return cat_name
