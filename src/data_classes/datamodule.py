from data_classes.dataset import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

class DataModule(LightningDataModule):
    def __init__(self, path_to_data, batch_size=24):
        super().__init__()
        self.batch_size = batch_size
        self.path_to_data = path_to_data

    def setup(self, stage=None):
        self.train_dataset = Dataset(self.path_to_data, 'train')
        self.val_dataset = Dataset(self.path_to_data, 'val')
        self.test_dataset = Dataset(self.path_to_data, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    

if __name__ == '__main__':
    data_module = DataModule(r'D:\MAGISTERKA\anomaly_traffic_road\datasets\DoTA')
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")