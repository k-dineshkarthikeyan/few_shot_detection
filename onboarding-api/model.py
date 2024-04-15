from torch import nn 
from torchvision.models import ResNet18_Weights
from torchvision import models, transforms
from torchvision import transforms

class SiameseNetwork(nn.Module):
    def __init__(self, batch_size=32, backbone='resnet18'):
        super(SiameseNetwork, self).__init__()

        self.batch_size = batch_size

        if backbone not in models.__dict__:
            raise Exception(f"No model named {backbone} exists in torchvision.models")
        self.backbone = models.__dict__[backbone](weights=ResNet18_Weights.DEFAULT, progress=True)
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
        self.transforms= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize([224,224], antialias=True)
        ])

    def forward(self, img ):
        img = self.transforms(img)
        img = img.unsqueeze(0)
        feature1 = self.backbone(img)
        return feature1
