from pytorch_lightning import callbacks
import torch
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import numpy as np
import os
from typing import Union, Tuple
import wandb

try:
    from country_region import *
    from data_processing import *
except Exception as e:
    print(e)
    

def fix_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

fix_all_seeds(42)


meta = pd.read_csv('Project/4000_PCS_human_origins/v44.3_HO_public.anno', sep='\t')
pcs = pd.read_csv('Project/4000_PCS_human_origins/pcs.txt', sep='\t')
    
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
            out_features=12  # Number of regions (in the dataset = 12, originally 14)  # 143 countries
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



INPUT_SIZE = 100
x, y, idx_to_country = prepare_data(pcs, meta, num_of_pcs=INPUT_SIZE)


def objective(trial):
    # Hyperparameters that is optimized
    batch_size = trial.suggest_int('batch_size', 12, 64)
    
    # Model inputs
    layer_1 = trial.suggest_int('layer_1', 100, 2000)
    layer_2 = trial.suggest_int('layer_2', 100, 2000)
    layer_3 = trial.suggest_int('layer_3', 100, 2000)
    layer_4 = trial.suggest_int('layer_4', 100, 2000)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    train_loader, val_loader = create_dataloaders(x, y, batch_size=batch_size, train_shuffle=True)
    
    model = ffnn(input_size=INPUT_SIZE, layer_1=layer_1, layer_2=layer_2, layer_3=layer_3, layer_4=layer_4,
                 dropout=dropout)
    
    gpu = 1 if torch.cuda.is_available() else 0
    
    print(f'Begin trial {trial.number}')
    # wandb_logger = WandbLogger(project='02456DeepLearning_project', name=f'input-{INPUT_SIZE}-{trial.number}')
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir='Project/lightning_logs', name=f'input_size_{INPUT_SIZE}-layer-4', version=trial.number),
        max_epochs=2,
        # log_every_n_steps=5,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val/acc')],
        gpus=gpu,
    )
    
    hyperparameters = dict(
        batch_size=batch_size,
        layer_1=layer_1,
        layer_2=layer_2,
        layer_3=layer_3,
        layer_4=layer_4,
        dropout=dropout
    )
    
    trainer.logger.log_hyperparams(hyperparameters)
    
    trainer.fit(model, train_loader, val_loader)
    
    # wandb_logger.experiment.finish()
    return trainer.callback_metrics['val/acc'].item()


wandb_kwargs = {"project": "02456DeepLearning_project", 'name': f'input-size-{INPUT_SIZE}-layer-4'}
wandbcb = WeightsAndBiasesCallback(metric_name='val/acc', wandb_kwargs=wandb_kwargs)

print('Finding optimal hyperparameters using Optuna...')
pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=50, callbacks=[wandbcb])
    
print('Best trial:')
trial = study.best_trial

print(f'\t Value: {trial.value}')
print(f'\t Params:')
for key, value in trial.params.items():
    print(f'\t \t {key}: {value}')
