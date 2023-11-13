# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv

# %%
class DCGNN(nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):         # k is the number of nearest neighbors
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.mlp   = MLP([3 * 64, 1024, 256, 128, out_channels], dropout=0.5, norm=None)
        
    def forward(self, data):                                # data: [x, pos, batch]
        x, pos, batch = data.x, data.pos, data.batch        # x: [N, C], pos: [N, 3], batch: [N]
        import pdb; pdb.set_trace()

        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))

        return F.log_softmax(out, dim=1)
        
        
# %% Test out the model using ShapeNet dataset and DataLoader
if __name__=='__main__':
    from paths import DATA
    from torch_geometric.datasets import ShapeNet
    from torch_geometric.data import DataLoader

    categories = ['Table', 'Lamp', 'Guitar', 'Motorbike']

    model       = DCGNN(out_channels=len(categories))
    dataset     = ShapeNet(root=DATA, categories=categories).shuffle()[:5000]
    dataloader  = DataLoader(dataset        = dataset,
                             batch_size     = 32,
                             shuffle        = True,
                             num_workers    = 12,
                             pin_memory     = True) # pin_memory=True to keep the data in GPU

    for batch in dataloader:
        out = model.forward(batch)
        break

    print(batch)
    print(batch.x.shape)
    print(batch.pos.shape)
    print(batch.batch.shape)
