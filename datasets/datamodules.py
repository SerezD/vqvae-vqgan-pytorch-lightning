import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.datasets import ImageDataset


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, train_folder: str, test_folder: str, image_size: int, batch_size: int, num_workers: int):
        """
        :param train_folder: the folder containing train images.
        :param test_folder: the folder containing test images.
        :param image_size: all images in the dataset will be resized like this (squared).
        :param batch_size: the same batch size is specified for every loader (train, test).
        :param num_workers: the same number of workers is specified for every loader (train, test).
        """

        super().__init__()

        self.train_folder = train_folder
        self.test_folder = test_folder

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.val = None

    def setup(self, stage=None):

        # prepare data
        self.train = ImageDataset(self.train_folder, self.image_size)
        self.val = ImageDataset(self.test_folder, self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False, drop_last=False)
