# %%
import random
from paths import DATA
import plotly.express as px
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T

# %% Plot 3D point cloud
def plot_3d_shape(shape):
    print('Number of data points: ', shape.pos.shape[0])
    x = shape.pos[:, 0]
    y = shape.pos[:, 1]
    z = shape.pos[:, 2]
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.3)
    fig.update_traces(marker_size=3)
    fig.show()

# %%
if __name__=='__main__':
    dataset_val   = ModelNet(root             = DATA,
                            train            = False,
                            name             = '10',
                            pre_transform    = T.NormalizeScale(),
                            transform        = T.SamplePoints(1024))
    sample_idx = random.choice(range(len(dataset_val)))
    plot_3d_shape(dataset_val[sample_idx])

    dataset_val[sample_idx]
