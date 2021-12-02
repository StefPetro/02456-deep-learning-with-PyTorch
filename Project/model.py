import torch
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

# Simple ffnn
class ffnn(pl.LightningModule):
    def __init__(self, input_size: int=100, layer_1: int=2000, layer_2: int=1500, layer_3: int=1000, layer_4: int=500,
                 dropout: float=0.25,
                 lr: float=1.5e-4, weight_decay: float=1e-4):
        super().__init__()
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.fc1 = nn.Linear(
            in_features=input_size,
            out_features=layer_1
        )

        self.fc2 = nn.Linear(
            in_features=layer_1,
            out_features=layer_2
        )
        
        self.fc3 = nn.Linear(
            in_features=layer_2,
            out_features=layer_3
        )
        
        self.fc4 = nn.Linear(
            in_features=layer_3,
            out_features=layer_4
        )

        self.fc_out = nn.Linear(
            in_features=layer_4,
            out_features=11  # Number of regions (in the dataset = 12, originally 14)  # 143 countries
        )

        self.dropout = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm1d(layer_1)
        self.batchnorm2 = nn.BatchNorm1d(layer_2)
        self.batchnorm3 = nn.BatchNorm1d(layer_3)
        self.batchnorm4 = nn.BatchNorm1d(layer_4)

    def forward(self, x):
        z = self.fc1(x)
        z = F.relu(z)
        z = self.batchnorm1(z)
        z = self.dropout(z)

        z = self.fc2(z)
        z = F.relu(z)
        z = self.batchnorm2(z)
        z = self.dropout(z)
        
        z = self.fc3(z)
        z = F.relu(z)
        z = self.batchnorm3(z)
        z = self.dropout(z)
        
        z = self.fc4(z)
        z = F.relu(z)
        z = self.batchnorm4(z)
        z = self.dropout(z)

        out = self.fc_out(z)

        return out
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        self.log('train/loss', loss, on_epoch=True)
        self.log('train/acc', acc, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)
        self.log('val/loss', loss, on_epoch=True)
        self.log('val/acc', acc, on_epoch=True)
        return loss
    