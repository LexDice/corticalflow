import os
import numpy as np
import pandas as pd


dataset_path = '/mnt/yaplab/data/junjiez/HCP_TRT'
list_sbj_name = np.array(os.listdir(dataset_path))
dataset_len = len(list_sbj_name)

# shuffle list
rng = np.random.default_rng(10)
list_indices = np.arange(dataset_len)
rng.shuffle(list_indices)
train, val, test = np.split(list_indices,
         [int(.75 * dataset_len), int(.85 * dataset_len)])

list_split_name = np.empty(dataset_len, dtype=object)
list_split_name[train] = 'train'
list_split_name[val] = 'val'
list_split_name[test] = 'test'

ds_split = pd.DataFrame({'subject': list_sbj_name, 'split': list_split_name})
ds_split.to_csv('./resources/hcp_split.csv', index=False)
