# %%
import os
import torch
import random

from paths import DATA
from train import TrainPointNet2

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.data import Data, Batch

from util import plot_3d_shape

# %%
CKPT_PATH = '/home/brian/github/DL_practice/PointNet2/model_checkpoint/aorus_20231116_114443/'
CKPT_FILENAME = 'epoch=14-loss=0.03583.ckpt' 
MODELNET_DATASET_ALIAS = '10'

# %%
trainer = TrainPointNet2.load_from_checkpoint(os.path.join(CKPT_PATH, CKPT_FILENAME), map_location=torch.device('cpu'))

dataset_val   = ModelNet(root             = DATA,
                         train            = False,
                         name             = MODELNET_DATASET_ALIAS,
                         pre_transform    = T.NormalizeScale(),
                         transform        = T.SamplePoints(1024))


val_dataloader = DataLoader(dataset        = dataset_val,
                            batch_size     = 16,
                            shuffle        = True,
                            num_workers    = 8,
                            pin_memory     = False)


# %% Plot a random 3D point cloud and print pred & actual
random_index = random.choice(range(len(dataset_val)))
print('='*90)
plot_3d_shape(dataset_val[random_index])

single_batch = Batch.from_data_list([dataset_val[random_index]])
out = trainer(single_batch)
pred = torch.argmax(out, dim=1)
actual = single_batch.y

print('Predicted:\t', pred)
print('Actual:\t\t', actual)


# %% Calculate accuracy
