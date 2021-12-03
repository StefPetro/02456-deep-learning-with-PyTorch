#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

with_ancient_col_acc = ['input-100-with_ancient-best_params - val/acc', 'input-20-with_ancient-best_params - val/acc', 'input-2-with_ancient-best_params - val/acc']
with_ancient_val_acc = pd.read_csv('wandb_csv/with_ancient_val_acc.csv')
with_ancient_col_loss = ['input-100-with_ancient-best_params - val/loss', 'input-20-with_ancient-best_params - val/loss', 'input-2-with_ancient-best_params - val/loss']
with_ancient_val_loss = pd.read_csv('wandb_csv/with_ancient_val_loss.csv')

no_ancient_col_acc = ['input-100-no_ancient-best_params - val/acc', 'input-20-no_ancient-best_params - val/acc', 'input-2-no_ancient-best_params - val/acc']
no_ancient_val_acc = pd.read_csv('wandb_csv/no_ancient_val_acc.csv')
no_ancient_col_loss = ['input-100-no_ancient-best_params - val/loss', 'input-20-no_ancient-best_params - val/loss', 'input-2-no_ancient-best_params - val/loss']
no_ancient_val_loss = pd.read_csv('wandb_csv/no_ancient_val_loss.csv')

epochs = np.arange(0, 200)

#%%
## With ancient samples
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for col_acc, col_loss in zip(with_ancient_col_acc, with_ancient_col_loss):
    split_list = col_acc.split('-')
    label = f"{split_list[1]} PCs"
    y1 = with_ancient_val_acc[col_acc].values
    axes[0].plot(epochs, y1, label=label)
    y2 = with_ancient_val_loss[col_loss].values
    axes[1].plot(epochs, y2, label=label)

axes[0].tick_params(labelsize=16)
axes[1].tick_params(labelsize=16)

axes[0].set_xlabel('Epochs', size=18)
axes[1].set_xlabel('Epochs', size=18)
axes[0].set_ylabel('Accuracy', size=18)
axes[1].set_ylabel('Loss', size=18)

axes[0].set_title('Accuracy for model with ancient samples', size=20)
axes[1].set_title('Loss for model with ancient samples', size=20)

axes[0].set_ylim(0.52, 1)
axes[1].set_ylim(0, 1.4)

axes[0].legend()
axes[1].legend()
plt.savefig('imgs/acc_loss_with_ancient.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

#%%
## Without ancient samples
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for col_acc, col_loss in zip(no_ancient_col_acc, no_ancient_col_loss):
    split_list = col_acc.split('-')
    label = f"{split_list[1]} PCs"
    y1 = no_ancient_val_acc[col_acc].values
    axes[0].plot(epochs, y1, label=label)
    y2 = no_ancient_val_loss[col_loss].values
    axes[1].plot(epochs, y2, label=label)

axes[0].tick_params(labelsize=16)
axes[1].tick_params(labelsize=16)

axes[0].set_xlabel('Epochs', size=18)
axes[1].set_xlabel('Epochs', size=18)
axes[0].set_ylabel('Accuracy', size=18)
axes[1].set_ylabel('Loss', size=18)

axes[0].set_title('Accuracy for model without ancient samples', size=20)
axes[1].set_title('Loss for model without ancient samples', size=20)

axes[0].set_ylim(0.52, 1)
axes[1].set_ylim(0, 1.4)

axes[0].legend()
axes[1].legend()
plt.savefig('imgs/acc_loss_no_ancient.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()
