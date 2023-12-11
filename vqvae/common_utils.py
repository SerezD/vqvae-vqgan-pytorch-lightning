import torch
import os
import yaml

from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, ToTorchImage

from ffcv_pl.data_loading import FFCVDataModule
from ffcv_pl.ffcv_utils.augmentations import DivideImage255
from ffcv_pl.ffcv_utils.utils import FFCVPipelineManager

from data.datamodules import ImageDataModule


def set_matmul_precision():
    """
    If using Ampere Gpus enable using tensor cores.
    Don't know exactly which other devices can benefit from this, but torch should throw a warning in case.
    Docs: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    """

    gpu_cores = os.popen('nvidia-smi -L').readlines()[0]

    if 'A100' in gpu_cores:
        torch.set_float32_matmul_precision('high')
        print('[INFO] set matmul precision "high"')


def get_model_conf(filepath: str):
    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def get_datamodule(loader_type: str, dirpath: str, image_size: int, batch_size: int, workers: int, seed: int,
                   is_dist: bool, mode: str = 'train'):

    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"dataset path not found: {dirpath}")

    else:

        if loader_type == 'standard':

            if mode == 'train':
                train_folder = f'{dirpath}train/'
                val_folder = f'{dirpath}validation/'
                return ImageDataModule(image_size, batch_size, workers, train_folder, val_folder)
            else:
                test_folder = f'{dirpath}test/'
                return ImageDataModule(image_size, batch_size, workers, test_folder=test_folder)

        elif loader_type == 'ffcv':

            if mode == 'train':

                train_manager = FFCVPipelineManager(
                    file_path=f'{dirpath}train.beton',
                    pipeline_transforms=[
                        [
                            CenterCropRGBImageDecoder((image_size, image_size), ratio=1),
                            ToTensor(),
                            ToTorchImage(),
                            DivideImage255(dtype=torch.float16)
                        ]
                    ],
                    ordering=OrderOption.RANDOM
                )

                val_manager = FFCVPipelineManager(
                    file_path=f'{dirpath}validation.beton',
                    pipeline_transforms=[
                        [
                            CenterCropRGBImageDecoder((image_size, image_size), ratio=1),
                            ToTensor(),
                            ToTorchImage(),
                            DivideImage255(dtype=torch.float16)
                        ]
                    ]
                )

                return FFCVDataModule(batch_size, workers, is_dist, train_manager, val_manager, seed=seed)

            else:
                test_manager = FFCVPipelineManager(
                    file_path=f'{dirpath}test.beton',
                    pipeline_transforms=[
                        [
                            CenterCropRGBImageDecoder((image_size, image_size), ratio=1),
                            ToTensor(),
                            ToTorchImage(),
                            DivideImage255(dtype=torch.float16)
                        ]
                    ]
                )

                return FFCVDataModule(batch_size, workers, is_dist, test_manager=test_manager, seed=seed)

        else:
            raise ValueError(f"loader type not recognized: {loader_type}")
