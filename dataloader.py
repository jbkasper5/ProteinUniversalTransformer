import sidechainnet as scn
import numpy as np
import pytorch_lightning as pl

class SCN_DataLoader(pl.LightningDataModule):
    def __init__(self):
        self.dataset = scn.load("debug", with_pytorch = 'dataloaders', batch_size = 12, dynamic_batching = True)
        self.max_len = max(max(list for list in [self.dataset[split].dataset.length for split in self.dataset.keys()]))



    # batch.masks returns the masks for the model to use
    def train_dataloader():
        pass

    def val_dataloader():
        pass

    def test_dataloader():
        pass