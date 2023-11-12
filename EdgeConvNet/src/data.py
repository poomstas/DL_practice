# %%
''' Make sure to set up the environment before running this script. See ../txt/requirements.txt '''
# Ref: https://wandb.ai/wandb/point-cloud-segmentation/reports/Digging-Into-the-ShapeNetCore-Dataset--VmlldzozMTY1ODA2

from torch_geometric.datasets import ShapeNet
from paths import DATA

dataset = ShapeNet(root=DATA, categories=['Table', 'Lamp', 'Guitar', 'Motorbike']).shuffle()[:5000]

# %%
if __name__=='__main__':
    import os
    import numpy as np
    from torch_geometric.data import DataLoader
    import torch_geometric.transforms as T

    categories = ['Table', 'Lamp', 'Guitar', 'Motorbike']
    n_dataset = 100

    # Load ShapeNet Dataset
    dataset = ShapeNet(root=DATA, categories=categories).shuffle()[:n_dataset]
    # train_dataloader = DataLoader(dataset        = dataset,
    #                               batch_size     = 32,
    #                               shuffle        = True,
    #                               num_workers    = 12,
    #                               pin_memory     = True) # pin_memory=True to keep the data in GPU

    random_index = np.random.randint(0, len(dataset))
    print(dataset[random_index])

# %% Applying transformations & normalization, and splitting the dataset
    transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()

    category="Airplane"
    dataset_path = os.path.join('ShapeNet', category)

    train_dataset = ShapeNet(
        dataset_path, category, split='trainval',
        transform=transform, pre_transform=pre_transform
    )
    val_dataset = ShapeNet(
        dataset_path, category, split='test',
        pre_transform=pre_transform
    )
