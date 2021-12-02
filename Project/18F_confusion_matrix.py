import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os

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

meta = pd.read_csv('Project/4000_PCS_human_origins/v44.3_HO_public.anno', sep='\t')
pcs = pd.read_csv('Project/4000_PCS_human_origins/pcs.txt', sep='\t')

x, y, idx_to_region = prepare_data(pcs, meta, num_of_pcs=INPUT_SIZE, ancient_samples=True)
_, _, test_loader = create_dataloaders(x, y, batch_size=batch_size, train_shuffle=True)

x, y_true = next(iter(test_loader))

## MODEL GOES HERE

y_pred = y_true

plt.figure(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, 
                                        normalize='true', display_labels=np.array(list(idx_to_region.keys())))

plt.xticks(rotation=45)
plt.show()
