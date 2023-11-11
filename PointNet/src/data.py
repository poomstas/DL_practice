# %%
''' Make sure to set up the environment before running this script. See ../txt/requirements.txt '''
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='./ShapeNet', categories=['Table', 'Lamp', 'Guitar', 'Motorbike']).shuffle()[:5000]

# %%
if __name__=='__main__':
    from torch_geometric.data import DataLoader
    from torch_geometric.datasets import ShapeNet
    from paths import DATA
    import numpy as np

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
