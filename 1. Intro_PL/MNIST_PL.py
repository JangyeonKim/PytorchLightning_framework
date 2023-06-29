import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

################# hyperparameters #################

lr = 1e-3
batch_size = 64
epochs = 10

################# dataset and dataloader #################

train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

train_data, val_data = random_split(train_data, [55000, 5000]) 

train_dataloader = DataLoader(train_data,batch_size=batch_size)
val_dataloader = DataLoader(val_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

################# Lightning Module #################

class NeuralNetwork(pl.LightningModule) :
    def __init__(self) :
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,10),
        )
        
    def forward(self, x) :
        return self.model(x)
    
    def training_step(self, batch, batch_idx) :
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def validation_step(self, batch, bath_idx) :
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics)
        
    def test_step(self, batch, bath_idx) : # validation 과 데이터만 다를 뿐 코드는 같음
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        metrics = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(metrics)
    
    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
    
################# training and test #################

model = NeuralNetwork()
trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=-1) # [0] : 0번 gpu, [1] : 1번 gpu, -1 : 가능한 모든 gpu

trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(dataloaders=test_dataloader)