import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
import numpy as np
import os
import yaml

try:
    from country_region import *
    from data_processing import *
    from model import ffnn
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


INPUT_SIZE = 20

# Load hyperparamters
filename = f'Project/best_hyperparameters/standard.yaml'  # f'Project/best_hyperparameters/input_{INPUT_SIZE}.yaml'
with open(filename, "r") as stream:
    hyperparameters = yaml.safe_load(stream)

batch_size = hyperparameters['batch_size']
layer_1    = hyperparameters['layer_1']
layer_2    = hyperparameters['layer_2']
layer_3    = hyperparameters['layer_3']
layer_4    = hyperparameters['layer_4']
dropout    = round(hyperparameters['dropout'], 5)

x, y, idx_to_country = prepare_data(pcs, meta, num_of_pcs=INPUT_SIZE, ancient_samples=True)
train_loader, val_loader, test_loader = create_dataloaders(x, y, batch_size=batch_size, train_shuffle=True)

gpu = 1 if torch.cuda.is_available() else 0

wandb_logger = WandbLogger(project='02456DeepLearning_project', name=f'input-{INPUT_SIZE}-with_ancient-standard_params')

trainer = pl.Trainer(
    logger=wandb_logger,
    # log_every_n_steps=5,
    gpus=gpu,
    max_epochs=100
)
    
trainer.logger.log_hyperparams(hyperparameters)

model = ffnn(input_size=INPUT_SIZE, layer_1=layer_1, layer_2=layer_2, layer_3=layer_3, layer_4=layer_4, dropout=dropout)

trainer.fit(model, train_loader, val_loader)

