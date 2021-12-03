import numpy as np
import pandas as pd
from typing import Union, Tuple
from pandas.core.reshape.merge import merge
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split 

from country_region import *


date_col = 'Full Date: One of two formats. (Format 1) 95.4% CI calibrated radiocarbon age (Conventional Radiocarbon Age BP, Lab number) e.g. 2624-2350 calBCE (3990±40 BP, Ua-35016). (Format 2) Archaeological context range, e.g. 2500-1700 BCE'


def prepare_data(pcs: pd.DataFrame, meta: pd.DataFrame, 
                 num_of_pcs: Union[None, int]=None, target: str='region', ancient_samples:bool=True) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    pcs: The principal components of the data
    meta: The meta data of the datam, as given by https://reichdata.hms.harvard.edu/pub/datasets/amh_repo/curated_releases/index_v44.3.html
    num_of_pcs: The number of principal components to include from the pcs dataframe. If None, all pcs are included.
    target: 2 options between region (12-14 categories) or country (143 categories)
    ancient_samples: if the data should include ancient samples (other samples than present). True = included.

    Output:
        x: is the features in a torch Tensor
        y: is the targets in a torch Tensor
        idx_to_country: a dictionary that maps the index in the target vector, to the representative country
    """
    
    if not ancient_samples:
        print('Removing ancient samples...')
        meta = meta[meta[date_col] == 'present']
    
    filter_meta = meta[['Version ID', 'Country']]

    merged = pd.merge(filter_meta, pcs, how='left', left_on=['Version ID'], right_on=['IID'])
    merged = merged[merged['Country'] != '..'].reset_index(drop=True) # first country is '..', so we remove that
    
    
    if target.lower() == 'country':
        countries = merged['Country'].unique()
        idx_to_target = dict(zip(countries, np.arange(len(countries))))
        merged['Country'] = merged.apply(lambda row: idx_to_target[row['Country']], axis=1)
        # one_hot_countries = pd.get_dummies(merged['Country']) 
        # y = torch.from_numpy(one_hot_countries.values)
        y = torch.from_numpy(merged['Country'].values)
    
    elif target.lower() == 'region':
        merged['Region'] = merged['Country'].map(country_region)
        merged = merged.dropna()
        regions = merged['Region'].unique()
        idx_to_target = dict(zip(regions, np.arange(len(regions))))
        #merged['Region'] = merged.apply(lambda row: idx_to_target[row['Region']], axis=1)
        merged['Region'] = merged['Region'].map(idx_to_target)
        # y = torch.from_numpy(one_hot_countries.values)
        y = torch.from_numpy(merged['Region'].values)
        
    x_all = merged.iloc[:, 4:].values

    if num_of_pcs is None:
        x = x_all
    else:
        x = x_all[:, :num_of_pcs]
    
    x = torch.from_numpy(x)
    
    # countries = one_hot_countries.columns
    # idx_to_country = dict(zip(np.arange(len(countries)), countries))

    return x.float(), y.long(), idx_to_target


def create_dataloaders(x:torch.Tensor, y:torch.Tensor, batch_size:int=64, train_shuffle:bool=False, save:bool=False):
    """
    x: Features which were created in the prepare_data function
    y: Targets which were created in the prepare_data function
    batch_size: Chooses the batch size of the dataloader
    save: If the dataloaders should be saved

    Output: Two dataloaders, one for training and another for validation
    """
    
    # First the full dataset is created from the tensors
    dataset = TensorDataset(x, y)

    # Then using random_split a test and val set is created
    train_size = int(0.7*len(dataset))
    val_size  = int(0.2*len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Then the dataloaders can initialized
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # test dataset is small enough to get whole dataset in one batch
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False) 

    if save:
        torch.save(train_loader, 'processed_data/train_loader.pth')
        torch.save(val_loader, 'processed_data/val_loader.pth')

    return train_loader, val_loader, test_loader

