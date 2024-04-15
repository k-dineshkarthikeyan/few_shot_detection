import lightning as L
import comet_ml
import torch
from torch import nn, optim
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dl import Coco
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F
import torchmetrics

train_path = "../one/train_2017"
test_path = "./val_2017"
train_annotation_path = './coco_train.json'
test_annotation_path = './coco_val.json'
workers =6
batch_size = 32
lr = 0.006
save_path = "./checkpoints/"
epochs=100

metrics = torchmetrics.Accuracy(task="multiclass", num_classes=2) #evaluation tool used to measure the accuracy


data_transforms = transforms.Compose([
                transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize([224,224], antialias=True)
            ])

train_ds = Coco(train_path, train_annotation_path, data_transforms)
val_ds = Coco(test_path, test_annotation_path, data_transforms)
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True, persistent_workers=True)
val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True, persistent_workers=True)

class Classifier(L.LightningModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1] )
        self.backbone.train()
        self.out = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,2)
            )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img1, img2):
        feature1 = self.backbone(img1)
        feature2 = self.backbone(img2)
        output = F.cosine_similarity(feature1, feature2).view(self.batch_size, 1)
        # output = output.unsqueeze(1)
        output = self.out(output)
        return output

    def training_step(self, batch, idx):
        x0, x1, y = batch
        print("x0 shape: ", x0.shape)
        print("x1 shape: ", x1.shape)
        print("y shape: ", y.shape)
        x = self(x0, x1)
        print("x shape: ", x.shape)
        loss = self.loss_fn(x, y)
        values = {'train_loss': loss}
        self.log_dict(values, prog_bar=True)
        self.logger.log_metrics(values)
        return loss
        
    def validation_step(self, batch, idx):
        x0, x1, y = batch
        x = self(x0, x1)
        loss = self.loss_fn(x,y)
        _, x = torch.max(x, dim=1)
        acc = metrics(x,y)
        values =  {'val_loss': loss, 'acc': acc}
        self.log_dict(values, prog_bar=True)
        self.logger.log_metrics(values)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=lr)

if __name__ == "__main__":
    comet_logger = CometLogger(api_key='pxJBRIzuAOeKXlVzYlKlgLC6k')
    model_checkpoint = ModelCheckpoint(save_path, save_top_k=-1, every_n_epochs=1)
    trainer = L.Trainer(log_every_n_steps=5, default_root_dir=save_path, max_epochs=epochs, min_epochs=10, callbacks=[model_checkpoint], logger=comet_logger)
    model = Classifier(batch_size = batch_size)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
