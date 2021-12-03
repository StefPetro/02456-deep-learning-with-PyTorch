import pytorch_lightning as pl

import pandas as pd
import numpy as np
import os
import yaml

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from pytorch_lightning.loggers import TensorBoardLogger

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

INPUT_SIZE = 2
BATCH_SIZE = 64
x, y, idx_to_country = prepare_data(pcs, meta, num_of_pcs=INPUT_SIZE)


def objective(trial):
    
    # Model inputs
    layer_1 = trial.suggest_int('layer_1', 100, 2000)
    layer_2 = trial.suggest_int('layer_2', 100, 2000)
    layer_3 = trial.suggest_int('layer_3', 100, 2000)
    layer_4 = trial.suggest_int('layer_4', 100, 2000)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    train_loader, val_loader, _ = create_dataloaders(x, y, batch_size=BATCH_SIZE, train_shuffle=True)
    
    model = ffnn(input_size=INPUT_SIZE, layer_1=layer_1, layer_2=layer_2, layer_3=layer_3, layer_4=layer_4,
                 dropout=dropout)
    
    gpu = 1 if torch.cuda.is_available() else 0
    
    print(f'Begin trial {trial.number}')
    # wandb_logger = WandbLogger(project='02456DeepLearning_project', name=f'input-{INPUT_SIZE}-{trial.number}')
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir='Project/lightning_logs', name=f'input_size_{INPUT_SIZE}-layer-4', version=trial.number),
        max_epochs=100,
        # log_every_n_steps=5,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val/acc')],
        gpus=gpu,
    )
    
    hyperparameters = dict(
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


with open(f'project/best_hyperparameters/input_{INPUT_SIZE}.yaml', 'w') as outfile:
    yaml.dump(trial.params, outfile, default_flow_style=False)
