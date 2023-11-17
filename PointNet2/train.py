# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_memlab import MemReporter
import socket
import wandb

from paths import DATA
from model import PointNet2
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader # https://github.com/Lightning-AI/lightning/issues/1557 -> Using torch geometric's DataLoader should work..
from torch_geometric.datasets import ModelNet
# from torch_geometric.data.lightning import LightningDataset # PyG's support for PyTorch Lightning. May have to use this

# %%
class TrainPointNet2(pl.LightningModule):
    ''' Train PointNet++ using PyTorch Lightning to classify ModelNet dataset '''
    def __init__(self,  
                 AUGMENTATIONS                  = T.SamplePoints(1024), # Need this to convert mesh into point cloud
                 LR                             = 0.001,
                 BATCH_SIZE                     = 128,
                 NUM_EPOCHS                     = 20,
                 N_DATASET                      = 5000,
                 MODELNET_DATASET_ALIAS         = '10', # 'ModelNet10' or 'ModelNet40'
                 STEP_LR_STEP_SIZE              = 20,
                 STEP_LR_GAMMA                  = 0.5,
                 SET_ABSTRACTION_RATIO_1        = 0.748,
                 SET_ABSTRACTION_RADIUS_1       = 0.4817,
                 SET_ABSTRACTION_RATIO_2        = 0.3316,
                 SET_ABSTRACTION_RADIUS_2       = 0.2447,
                 DROPOUT                        = 0.1,
                 ):

        super(TrainPointNet2, self).__init__()

        self.save_hyperparameters()             # Need this later to load_from_checkpoint without providing the hyperparams again

        self.augmentations                      = AUGMENTATIONS
        self.lr                                 = LR
        self.bs                                 = BATCH_SIZE
        self.n_epochs                           = NUM_EPOCHS
        self.n_dataset                          = N_DATASET
        self.modelnet_dataset_alias             = MODELNET_DATASET_ALIAS

        self.step_lr_step_size                  = STEP_LR_STEP_SIZE
        self.step_lr_gamma                      = STEP_LR_GAMMA

        self.set_abstraction_ratio_1            = SET_ABSTRACTION_RATIO_1
        self.set_abstraction_radius_1           = SET_ABSTRACTION_RADIUS_1
        self.set_abstraction_ratio_2            = SET_ABSTRACTION_RATIO_2
        self.set_abstraction_radius_2           = SET_ABSTRACTION_RADIUS_2
        self.dropout                            = DROPOUT


        self.model                  = PointNet2(set_abstraction_ratio_1   = self.set_abstraction_ratio_1, 
                                                set_abstraction_ratio_2   = self.set_abstraction_ratio_2,
                                                set_abstraction_radius_1  = self.set_abstraction_radius_1,
                                                set_abstraction_radius_2  = self.set_abstraction_radius_2,
                                                dropout                   = self.dropout, 
                                                n_classes                 = 10)

        # self.loss                               = F.nll_loss  # Functional form. The model itself returns log_softmax, so we use NLL Loss instead of nn.CrossEntropyLoss()
        self.loss                               = torch.nn.NLLLoss() # Class form. The model itself returns log_softmax, so we use NLL Loss instead of nn.CrossEntropyLoss()
        self.loss_cum                           = 0

        print('='*90)
        print('MODEL HYPERPARAMETERS')
        print('='*90)
        print(self.hparams)
        print('='*90)


    def setup(self, stage:str):
        # self.reporter = MemReporter(model) # Set up memory reporter
        # self.reporter.report()
        return


    def prepare_data(self):
        ModelNet(root=DATA,  train=True, name=self.modelnet_dataset_alias, pre_transform=T.NormalizeScale()) # Specify 10 or 40 (ModelNet10, ModelNet40)

        self.dataset_train = ModelNet(
                root             = DATA,
                train            = True,
                name             = self.modelnet_dataset_alias,
                pre_transform    = T.NormalizeScale(),
                transform        = self.augmentations)

        self.dataset_val   = ModelNet(
                root             = DATA,
                train            = False,
                name             = self.modelnet_dataset_alias,
                pre_transform    = T.NormalizeScale(),
                transform        = self.augmentations)
        
        # self.lightning_dataset = LightningDataset(dataset_train, dataset_val) # PyG's support for PyTorch Lightning
        
    def train_dataloader(self):
        train_dataloader = DataLoader(dataset        = self.dataset_train,
                                      batch_size     = self.bs,
                                      shuffle         = True,
                                      num_workers    = 8,
                                      pin_memory     = False) # pin_memory=True to keep the data in GPU
        return train_dataloader
        
        
    def val_dataloader(self):
        val_dataloader   = DataLoader(dataset        = self.dataset_val,
                                      batch_size     = self.bs,
                                      shuffle         = False,
                                      num_workers    = 8,
                                      pin_memory     = False) # pin_memory=True to keep the data in GPU
        return val_dataloader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size   = self.step_lr_step_size,
                                                    gamma       = self.step_lr_gamma)
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]


    def forward(self, data):
        return self.model(data)


    def training_step(self, batch, batch_idx):
        pred, target = self.forward(batch), batch.y
        loss = self.loss(pred, target)
        self.loss_cum += loss.item() # Not including the .item() will cause the loss to accumulate in GPU memory
        self.log('loss', loss)
        return {'loss': loss}


    def on_train_epoch_end(self):
        self.log('loss_epoch', self.loss_cum)
        self.loss_cum = 0
        # self.reporter.report() # Report memory usage
        return {'loss_epoch': self.loss_cum}
    

    def validation_step(self, batch, batch_idx):
        pred, target = self.forward(batch), batch.y
        loss = self.loss(pred, target)
        self.log('val_loss', loss)
        return {'val_loss': loss}


    # def test_step(self, batch, batch_idx):
    #     pred, target = self.forward(batch), batch.y
    #     loss = self.loss(pred, target)
    #     self.log('test_loss', loss)

# %%
if __name__=='__main__':
    torch.set_float32_matmul_precision('medium') # medium or high. See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    hostname = socket.gethostname()
    run_ID = '_'.join([hostname, datetime.now().strftime('%Y%m%d_%H%M%S')])
    print('Hostname: {}'.format(hostname))

    cb_checkpoint = ModelCheckpoint(dirpath     = './model_checkpoint/{}/'.format(run_ID),
                                    monitor     = 'val_loss',
                                    filename    = '{val_loss:.5f}-{loss:.5f}-{epoch:02d}',
                                    save_top_k  = 10)

    trainer = Trainer(
        max_epochs                      = 30,
        accelerator                     = 'gpu',  # set to cpu to address CUDA errors.
        strategy                        = 'auto', # Currently only the pytorch_lightning.strategies.SingleDeviceStrategy and pytorch_lightning.strategies.DDPStrategy training strategies of  PyTorch Lightning are supported in order to correctly share data across all devices/processes
        # devices                         = 'auto',    # [0, 1] or use 'auto'
        devices                         = [1],    # [0, 1] or use 'auto'
        log_every_n_steps               = 1,
        fast_dev_run                    = False,     # Run a single-batch through train & val and see if the code works
        logger                          = [],
        callbacks                       = [cb_checkpoint])

    model = TrainPointNet2()

    trainer.fit(model)
