#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
import yaml 

from torch.nn import functional as F
from model import *

from data_processing import *

def fix_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

fix_all_seeds(42)


INPUT_SIZE = 20
batch_size = 64

meta = pd.read_csv('4000_PCS_human_origins/v44.3_HO_public.anno', sep='\t')
pcs = pd.read_csv('4000_PCS_human_origins/pcs.txt', sep='\t')

x, y, idx_to_region = prepare_data(pcs, meta, num_of_pcs=INPUT_SIZE, ancient_samples=True)
_, _, test_loader = create_dataloaders(x, y, batch_size=batch_size, train_shuffle=True)

x, y_true = next(iter(test_loader))

# Load hyperparamters
filename = f'best_hyperparameters/input_{INPUT_SIZE}.yaml'  # f'Project/best_hyperparameters/standard.yaml'  
with open(filename, "r") as stream:
    hyperparameters = yaml.safe_load(stream)

layer_1    = hyperparameters['layer_1']
layer_2    = hyperparameters['layer_2']
layer_3    = hyperparameters['layer_3']
layer_4    = hyperparameters['layer_4']
dropout    = round(hyperparameters['dropout'], 5)

model_file = f'checkpoints\input_{INPUT_SIZE}/with_ancient_epoch=197_val_acc=0.00.ckpt'

model = ffnn.load_from_checkpoint(model_file, input_size=INPUT_SIZE, layer_1=layer_1, layer_2=layer_2, layer_3=layer_3, layer_4=layer_4, dropout=dropout)
model.eval()

logits = model(x)
y_pred = logits.max(1).indices  # Get the indices (class) with largest probability

#%%
region_idx = {val: key for key, val in idx_to_region.items()}
regions = np.sort(list(idx_to_region.keys()))
new_region_to_idx = dict(zip(regions, np.arange(11)))
new_y_true = torch.tensor([new_region_to_idx[region_idx[y.item()]] for y in y_true])
new_y_pred = torch.tensor([new_region_to_idx[region_idx[y.item()]] for y in y_pred])


#%%
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(
    new_y_true.detach().numpy(), new_y_pred.detach().numpy(), 
    normalize='true', 
    display_labels=np.array(list(new_region_to_idx.keys())),
    include_values=False,
    ax=ax
    )

plt.xticks(rotation=90, size=24)
plt.yticks(size=24)

plt.xlabel('Predicted label', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.savefig('imgs/confusionmatrix_with_ancient.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# %%
