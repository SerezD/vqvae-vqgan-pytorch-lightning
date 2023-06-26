# VQ-VAE/GAN Pytorch Lightning

Pytorch lightning implementation of both VQVAE/VQGAN, with different quantization algorithms.
Uses [FFCV](https://github.com/libffcv/ffcv) for fast data loading and [WandB](https://github.com/wandb/wandb)
for logging.

### Citations

Original vqvae paper: https://arxiv.org/abs/1711.00937  
Original vqgan paper: https://arxiv.org/abs/2012.09841

Original vqvae code: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py  
Original vqgan code: https://github.com/CompVis/taming-transformers

Some architectural improvements are taken by:  
MaskGit: 
   - paper https://arxiv.org/abs/2202.04200
   - code https://github.com/google-research/maskgit  

Improved VQGAN: https://arxiv.org/abs/2110.04627

Perceptual Loss part cloned from: https://github.com/S-aiueo32/lpips-pytorch/tree/master

Discriminator cloned and modified from:  https://github.com/rosinality/stylegan2-pytorch/blob/master  
Discriminator Losses (hinge / non-saturating): https://github.com/google-research/maskgit

Quantization Algorithms: 
   - Standard and EMA update: Original VQVAE paper. 
   - Gumbel Softmax: https://github.com/karpathy/deep-vector-quantization
   - "Entropy" Quantizer: https://github.com/google-research/maskgit

Fast Data Loading:
   - FFCV: https://github.com/libffcv/ffcv
   - FFCV_PL: https://github.com/SerezD/ffcv_pytorch_lightning


### Installation

For fast solving, I suggest to use libmamba:  
https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community

```
# Dependencies Install 
conda env create --file environment.yml
conda activate vqvae

# package install (after cloning)
pip install .
```

### Datasets and DataLoaders

This repository allows for both fast (`FFCV`) and standard pytorch-lightning data loading.

In each case, your dataset can be composed of images in `.png .jpg .bmp .JPEG` formats.  
The dataset structure must be like the following:
 ```
  🗂 path/to/dataset/
     📂 train/
      ┣ 000 .jpeg
      ┣ 001.jpg
      ┗ 002.png
     📂 validation/
      ┣ 003.jpeg
      ┣ 004.jpg
      ┗ 005.png
 ```

If you want to use `FFCV`, you must first create the `.beton` files. For this you can use the `create_beton_file.py` script
int the `/data` directory.

```
# example
# creates 2 beton files (one for val and one for training) 
# in the /home/datasets/examples/beton_dataset directory. 
# the max resolution of the preprocessed images will be 256x256

python ./data/create_beton_file.py --max_resolution 256 /
                                   --output_folder "/home/datasets/examples/beton_dataset" /
                                   --train_folder "/home/datasets/examples/train" /
                                   --val_folder "/home/datasets/examples/validation"
```

For more information on fast loading, check:
   - FFCV: https://github.com/libffcv/ffcv
   - FFCV_PL: https://github.com/SerezD/ffcv_pytorch_lightning


### Configuration Files

The configuration files `.yaml` will provide all the details on the type of autoencoder that
you want to train. 
Complete examples are in the `/example_confs/` directory:
   - `standard_vqvae`: train a base vqvae with standard quantization algorithm.
   - `standard_vqgan`: train a vqgan model with standard quantization algorithm.
   - `ema_vqgan`: train a vqgan model with EMA variant of the quantization algorithm.
   - `gumbel_vqgan`: train a vqgan model with Gumbel-Softmax quantization algorithm.
   - `entropy_vqgan`: train a vqgan model with entropy loss quantization algorithm.

### Training

Once dataset and configuration file are created, run training script like:  
```
  python ./vqvae/train.py --params_file "./example_confs/standard_vqvae.yaml" \
                          --dataloader standard \  # uses standard PL data-loader
                          --dataset_path "/home/datasets/examples/" \ # contains train/validation folders
                          --save_path "./runs/" \ 
                          --run_name vqvae_standard_quantization \
                          --seed 1234 \  # fix seed for reproducibility
                          --logging \  # will log results to wandb
                          --workers 8               
```