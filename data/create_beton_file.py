import argparse
import os

from ffcv.fields import RGBImageField
from ffcv_pl.generate_dataset import create_beton_wrapper

from data.datasets import ImageDataset


def get_args():

    parser = argparse.ArgumentParser(
        description='Define an Image Dataset using ffcv for fast data loading')

    parser.add_argument('--max_resolution', type=int, default=256)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--train_folder', type=str, default=None)
    parser.add_argument('--val_folder', type=str, default=None)
    parser.add_argument('--test_folder', type=str, default=None)
    parser.add_argument('--predict_folder', type=str, default=None)

    return parser.parse_args()


def main():

    args = get_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.train_folder is not None:
        train_dataset = ImageDataset(folder=args.train_folder, image_size=args.max_resolution, ffcv=True)

        # https://docs.ffcv.io/api/fields.html
        fields = (RGBImageField(write_mode='jpg', max_resolution=args.max_resolution),)
        create_beton_wrapper(train_dataset, f"{args.output_folder}/train.beton", fields=fields)

    if args.val_folder is not None:
        val_dataset = ImageDataset(folder=args.val_folder, image_size=args.max_resolution, ffcv=True)

        # https://docs.ffcv.io/api/fields.html
        fields = (RGBImageField(write_mode='jpg', max_resolution=args.max_resolution),)
        create_beton_wrapper(val_dataset, f"{args.output_folder}/validation.beton", fields=fields)

    if args.test_folder is not None:
        test_dataset = ImageDataset(folder=args.test_folder, image_size=args.max_resolution, ffcv=True)

        # https://docs.ffcv.io/api/fields.html
        fields = (RGBImageField(write_mode='jpg', max_resolution=args.max_resolution),)
        create_beton_wrapper(test_dataset, f"{args.output_folder}/test.beton", fields=fields)

    if args.predict_folder is not None:
        predict_dataset = ImageDataset(folder=args.predict_folder, image_size=args.max_resolution, ffcv=True)

        # https://docs.ffcv.io/api/fields.html
        fields = (RGBImageField(write_mode='jpg', max_resolution=args.max_resolution),)
        create_beton_wrapper(predict_dataset, f"{args.output_folder}/predict.beton", fields=fields)


if __name__ == '__main__':

    main()
