from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from glob import glob


class ImageDataset(Dataset):

    def __init__(self, folder: str, image_size: int):
        """
        :param folder: path to images
        :param image_size: size for images (squared)
        """

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                                             transforms.Resize((image_size, image_size)),
                                             ])
        self.image_size = image_size

        self.image_names = glob(folder + '*.png') + glob(folder + '*.jpg') + \
                           glob(folder + '*.bmp') + glob(folder + '*.JPEG')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        """
        :param index: the image index to return
        :return: the corresponding image and optionally class name
        """

        # load image
        image_tensor = self.transform(Image.open(self.image_names[index]).convert('RGB'))

        return image_tensor
