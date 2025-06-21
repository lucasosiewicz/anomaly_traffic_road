from data_classes.unsupervised_dataset import UnsupervisedDataset
from data_classes.supervised_dataset import SupervisedDataset
from data_classes.video_dataset import VideoDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from math import floor
import numpy as np
import psutil
import os

def calculate_dataloader_params(batch_size, img_size=(227, 227), image_channels=3, sequence_length=16, precision=32, ram_fraction=0.8):
    """
    Function calculates the number of workers and prefetch factor
    for DataLoader based on the available RAM.

    Input:
        batch_size: int - the batch size used in DataLoader
        img_size: tuple - the size of the image (height, width)
        image_channels: int - the number of channels in the image
        sequence_length: int - the length of video sequence
        precision: int - the precision of the weights
        ram_fraction: float - the fraction of RAM to use
    Output:
        dict of params: num_workers, prefetch_factor, pin_memory, persistent_workers
    """
    
    total_ram = psutil.virtual_memory().available * ram_fraction
    img_memory = np.prod(img_size) * image_channels * (precision/8)
    sequence_memory = sequence_length * img_memory
    batch_memory = batch_size * sequence_memory

    if batch_memory > total_ram:
        raise ValueError("Batch size too large for available RAM. Reduce the batch size or sequence length.")

    max_batches_in_ram = floor(total_ram / batch_memory)

    prefetch_factor = min(max_batches_in_ram, 16)
    num_workers = min(floor(prefetch_factor / 2), os.cpu_count())

    params = {"num_workers": num_workers,
              "prefetch_factor": prefetch_factor,
              "pin_memory": True,
              "persistent_workers": True}

    return params

class DataModule(LightningDataModule):
    def __init__(
        self, 
        path_to_data, 
        dataset,
        batch_size=24, 
        unsupervised=True,
        is_sequence=False,
        sequence_length=16,
        stride=4,
        transform=None,
        target_transform=None,
        crop_type=None,
        target_class='all'
    ):
        super().__init__()
        self.batch_size = batch_size
        self.path_to_data = path_to_data
        self.dataset = dataset
        self.unsupervised = unsupervised
        self.is_sequence = is_sequence
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.target_transform = target_transform
        self.crop_type = crop_type
        self.target_class = target_class
        
        # Parametry dla DataLoadera bez wielowątkowości
        self.params = calculate_dataloader_params(
            batch_size=self.batch_size,
            img_size=(227, 227),
            image_channels=1,
            sequence_length=self.sequence_length,
            precision=32,
            ram_fraction=0.8
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.unsupervised:
            self.train_dataset = UnsupervisedDataset(
                self.path_to_data, 
                'train', 
                resize_target=(227, 227), 
                crop_type=self.crop_type,
                dataset_name=self.dataset
            )
            self.val_dataset = UnsupervisedDataset(
                self.path_to_data, 
                'val', 
                resize_target=(227, 227), 
                crop_type=self.crop_type,
                dataset_name=self.dataset
            )
            self.test_dataset = UnsupervisedDataset(
                self.path_to_data, 
                'test', 
                resize_target=(227, 227), 
                crop_type=self.crop_type,
                dataset_name=self.dataset
            )
        else:
            if self.is_sequence:
                self.train_dataset = VideoDataset(
                    self.path_to_data,
                    which_set='train',
                    sequence_length=self.sequence_length,
                    stride=self.stride,
                    transform=self.transform,
                    target_transform=self.target_transform
                )
                self.val_dataset = VideoDataset(
                    self.path_to_data,
                    which_set='val',
                    sequence_length=self.sequence_length,
                    stride=self.stride,
                    transform=self.transform,
                    target_transform=self.target_transform
                )
                self.test_dataset = VideoDataset(
                    self.path_to_data,
                    which_set='test',
                    sequence_length=self.sequence_length,
                    stride=self.stride,
                    transform=self.transform,
                    target_transform=self.target_transform
                )
            else:
                self.train_dataset = SupervisedDataset(
                    self.path_to_data, 
                    which_set='train', 
                    target_class=self.target_class, 
                    crop_type=self.crop_type,
                    dataset_name=self.dataset
                )
                self.val_dataset = SupervisedDataset(
                    self.path_to_data, 
                    which_set='val', 
                    target_class=self.target_class, 
                    crop_type=self.crop_type,
                    dataset_name=self.dataset
                )
                self.test_dataset = SupervisedDataset(
                    self.path_to_data, 
                    which_set='test', 
                    target_class=self.target_class, 
                    crop_type=self.crop_type,
                    dataset_name=self.dataset
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=not self.is_sequence,  # Wyłączamy shuffle dla sekwencji
            **self.params
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            **self.params
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            **self.params
        )
