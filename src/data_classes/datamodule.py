from lightning import LightningDataModule

class DataModule(LightningDataModule):
    def __init__(self, batch_size=24):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass