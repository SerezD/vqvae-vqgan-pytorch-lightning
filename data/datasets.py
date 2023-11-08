import pathlib
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset

from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, folder: str, image_size: int, ffcv: bool = False):

        self.samples = sorted(list(pathlib.Path(folder).rglob('*.png')) + list(pathlib.Path(folder).rglob('*.jpg')) +
                              list(pathlib.Path(folder).rglob('*.bmp')) + list(pathlib.Path(folder).rglob('*.JPEG')))
        self.ffcv = ffcv

        self.transforms = Compose([ToTensor(), Resize((image_size, image_size), antialias=True)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # path to string
        image_path = self.samples[idx].absolute().as_posix()

        image = Image.open(image_path).convert('RGB')

        return (image, ) if self.ffcv else self.transforms(image)
