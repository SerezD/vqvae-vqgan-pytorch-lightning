import argparse
import pytorch_lightning as pl

from vqvae.common_utils import get_model_conf, get_datamodule
from vqvae.model import VQVAE


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--params_file', type=str, required=True, help='path to yaml file with model params')
    parser.add_argument('--dataloader', type=str, choices=['standard', 'ffcv'], default='standard',
                        help='defines what type of dataloader to use.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='path to a dataset folder containing two sub-folders (validation / train) or beton files '
                             '(train.beton / validation.beton).')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='evaluation is on onr gpu, set batch size.')
    parser.add_argument('--seed', type=int, required=True, help='global random seed for reproducibility')
    parser.add_argument('--loading_path', type=str, required=True,
                        help='path to checkpoint to load')
    parser.add_argument('--workers', type=int, help='num of parallel workers', default=1)

    return parser.parse_args()


def main():

    args = parse_args()
    conf = get_model_conf(args.params_file)

    # configuration params (assumes some env variables in case of multi-node setup)
    workers = int(args.workers)
    seed = int(args.seed)

    batch_size = args.batch_size

    pl.seed_everything(seed, workers=True)

    load_checkpoint_path = args.loading_path

    # model params
    image_size = int(conf['image_size'])
    ae_conf = conf['autoencoder']
    q_conf = conf['quantizer']

    # get model
    model = VQVAE.load_from_checkpoint(load_checkpoint_path, strict=False, image_size=image_size, ae_conf=ae_conf,
                                       q_conf=q_conf, l_conf=None, t_conf=None, init_cb=False, load_loss=False)

    # data loading (standard pytorch lightning or ffcv)
    datamodule = get_datamodule(args.dataloader, args.dataset_path, image_size, batch_size,
                                workers, seed, is_dist=False, mode='test')

    # trainer
    trainer = pl.Trainer(strategy='ddp', accelerator='gpu', devices=1, precision='16-mixed', deterministic=True)
    print(f"[INFO] workers: {workers}")
    print(f"[INFO] batch size: {batch_size}")

    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':

    main()
