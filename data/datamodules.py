import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.datasets import ImageDataset


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, image_size: int, batch_size: int, num_workers: int, train_folder: str = None,
                 val_folder: str = None, test_folder: str = None, predict_folder: str = None):
        """
        :param image_size: all images in the dataset will be resized like this (squared).
        :param batch_size: the same batch size is specified for every loader (train, test).
        :param num_workers: the same number of workers is specified for every loader (train, test).
        :param train_folder: the folder containing train images.
        :param test_folder: the folder containing test images.
        """

        super().__init__()

        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder
        self.predict_folder = predict_folder

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None
        self.predict = None

    def setup(self, stage: str):

        if stage == 'fit':
            if self.train_folder is not None:
                self.train = ImageDataset(self.train_folder, self.image_size, ffcv=False)
            if self.val_folder is not None:
                self.val = ImageDataset(self.val_folder, self.image_size, ffcv=False)
        elif stage == 'validate':
            if self.val_folder is not None:
                self.val = ImageDataset(self.val_folder, self.image_size, ffcv=False)
        elif stage == 'test':
            if self.test_folder is not None:
                self.test = ImageDataset(self.test_folder, self.image_size, ffcv=False)
        elif stage == 'predict':
            if self.predict_folder is not None:
                self.predict = ImageDataset(self.predict_folder, self.image_size, ffcv=False)
        else:
            pass

    def train_dataloader(self):
        if self.train is None:
            pass
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        if self.val is None:
            pass
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False, drop_last=False)

    def test_dataloader(self):
        if self.test is None:
            pass
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False, drop_last=False)

    def predict_dataloader(self):
        if self.predict is None:
            pass
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False, drop_last=False)
