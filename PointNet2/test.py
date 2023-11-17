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

from util import plot_3d_shape, plot_confusion_matrix
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %%
CKPT_PATH = '/home/brian/github/DL_practice/PointNet2/model_checkpoint/aorus_20231116_114443/'
CKPT_FILENAME = 'epoch=14-loss=0.03583.ckpt' 
MODELNET_DATASET_ALIAS = '10'
CLASSES_MODELNET_10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'] # ModelNet10 classes TODO Check this!

# %%
trainer = TrainPointNet2.load_from_checkpoint(os.path.join(CKPT_PATH, CKPT_FILENAME), map_location=torch.device('cpu'))

dataset_val   = ModelNet(root             = DATA,
                         train            = False,
                         name             = MODELNET_DATASET_ALIAS,
                         pre_transform    = T.NormalizeScale(),
                         transform        = T.SamplePoints(1024))


val_dataloader = DataLoader(dataset        = dataset_val,
                            batch_size     = 128,
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

print('Predicted:\t', CLASSES_MODELNET_10[pred.item()])
print('Actual:\t\t', CLASSES_MODELNET_10[actual.item()])


# %% Calculate classification metrics
targets, preds = [], []
for batch in tqdm(val_dataloader):
    actual = batch.y
    pred = torch.argmax(trainer(batch), dim=1)
    targets.append(actual)
    preds.append(pred)

targets = torch.cat(targets)
preds = torch.cat(preds)

# %%
y_true = targets.numpy()
y_pred = preds.numpy()

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')  
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

conf_matrix = confusion_matrix(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("Confusion Matrix:\n", conf_matrix)

plot_confusion_matrix(conf_matrix, classes=CLASSES_MODELNET_10, figsize=(5,5), text_size=10)
