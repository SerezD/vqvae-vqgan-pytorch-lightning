from abc import ABC, abstractmethod
import torch
from kornia.enhance import normalize, denormalize
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomResizedCrop


class BaseVQVAE(ABC):
    """
    Defines the methods that can be used to work with indices in two stage models,
    as well as preprocessing
    """

    def __init__(self, image_size: int):

        super().__init__()

        self.image_size = image_size

        self.training_augmentations = AugmentationSequential(RandomResizedCrop((image_size, image_size)),
                                                             RandomHorizontalFlip(),
                                                             same_on_batch=False)

        # init values for child classes (may never be implemented)
        self.scheduler = None

        # codebook usage counts for re-init (train) or logging (validation) (may never be used)
        self.train_epoch_usage_count = None
        self.val_epoch_usage_count = None

    @torch.no_grad()
    def preprocess_batch(self, images: torch.Tensor, mean: tuple = (0.485, 0.456, 0.406),
                         std: tuple = (0.229, 0.224, 0.225), training: bool = False):
        """
        :param images: batch of float32 tensors B, C, H, W assumed in range 0__1.
        :param mean: 3 channels mean vector for normalization, default to imagenet values
        :param std: 3 channels std vector for normalization, default to imagenet values
        :param training: if True, additionally applies self.training_augmentations
                        (default to Random HFlip and Random Resized Crop)
        :return normalized images B, C, H, W, ready for the forward pass.
        """

        # ensure 0 _ 1 values (no effect if correctly loaded)
        images = torch.clamp(images, 0., 1.)

        # training data augmentation
        if training:
            images = self.training_augmentations(images)

        images = normalize(images, torch.tensor(mean), torch.tensor(std))
        return images

    @torch.no_grad()
    def preprocess_visualization(self, images: torch.Tensor, mean: tuple = (0.485, 0.456, 0.406),
                                 std: tuple = (0.229, 0.224, 0.225)):
        """
        Preprocess images by de-normalizing back to range 0_1
        :param images: (B C H W) output of the autoencoder
        :param mean: 3 channels mean vector for de-normalization, default to imagenet values
        :param std: 3 channels std vector for de-normalization, default to imagenet values
        :return denormalized images in range 0__1
        """
        images = denormalize(images, torch.tensor(mean), torch.tensor(std))
        images = torch.clip(images, 0, 1)  # if mean,std are correct, should have no effect
        return images

    @abstractmethod
    def get_tokens(self, images: torch.Tensor) -> torch.IntTensor:
        """
        :param images: B, 3, H, W
        :return B, S batch of codebook indices
        """
        pass

    @abstractmethod
    def quantize(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: B, 3, H, W
        :return B, S, D batch of quantized
        """
        pass

    @abstractmethod
    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: B, 3, H, W
        :return reconstructions (B, 3, H, W)
        """
        pass

    @abstractmethod
    def reconstruct_from_tokens(self, tokens: torch.IntTensor) -> torch.Tensor:
        """
        :param tokens: B, S where S is the sequence len
        :return (B, 3, H, W) reconstructed images
        """
        pass
