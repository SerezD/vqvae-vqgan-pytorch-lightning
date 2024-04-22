# VQ-VAE/GAN Pytorch Lightning

Pytorch lightning implementation of both VQVAE/VQGAN, with different quantization algorithms.
Uses [FFCV](https://github.com/libffcv/ffcv) for fast data loading and [WandB](https://github.com/wandb/wandb)
for logging.

### Acknowledgments and Citations

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

Discriminator cloned from:  https://github.com/NVlabs/stylegan2-ada-pytorch  
Discriminator Losses (hinge / non-saturating): https://github.com/google-research/maskgit

Quantization Algorithms: 
   - Standard and EMA update: Original VQVAE paper. 
   - Gumbel Softmax: code taken from https://github.com/karpathy/deep-vector-quantization, 
     parameters from DALL-E paper: https://arxiv.org/abs/2102.12092. Also check: https://arxiv.org/abs/1611.01144
     for a theoretical understanding.
   - "Entropy" Quantizer: code taken from https://github.com/google-research/maskgit

Fast Data Loading:
   - FFCV: https://github.com/libffcv/ffcv
   - FFCV_PL: https://github.com/SerezD/ffcv_pytorch_lightning

### Installation

For fast solving, I suggest to use libmamba:  
https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community

*Note: Check the `pytorch-cuda` version in `environment.yml` to ensure it is compatible with your cuda version.*

```
# Dependencies Install 
conda env create --file environment.yml
conda activate vqvae

# package install (after cloning)
pip install .
```

#### Stylegan custom ops

StyleGan discriminator uses custom cuda operations, written by the NVIDIA team to speed up training.   
This requires to install NVIDIA-CUDA TOOLKIT: https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md

In this repo, instead of NVIDIA-CUDA TOOLKIT, the `environment.yml` installs: https://anaconda.org/conda-forge/cudatoolkit-dev  
I found this to be an easier option, and apparently everything works fine.

### Datasets and DataLoaders

This repository allows for both fast (`FFCV`) and standard (`pytorch`) data loading.

In each case, your dataset can be composed of images in `.png .jpg .bmp .JPEG` formats.  
The dataset structure must be like the following:
 ```
  🗂 path/to/dataset/
     📂 train/
      ┣ 000.jpeg
      ┣ 001.jpg
      ┗ 002.png
     📂 validation/
      ┣ 003.jpeg
      ┣ 004.bmp
      ┗ 005.png
     📂 test/
      ┣ 006.jpeg
      ┣ 007.jpg
      ┗ 008.bmp
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

The configuration file `.yaml` provides all the details on the type of autoencoder that
you want to train (check the folder "./example_confs").

### Training

Once dataset and configuration file are created, run training script like:  
```
  python ./vqvae/train.py --params_file "./example_confs/standard_vqvae_cb1024.yaml" \
                          --dataloader ffcv \  # uses ffcv data-loader
                          --dataset_path "/home/datasets/examples/" \ # contains train/validation .beton file
                          --save_path "./runs/" \ 
                          --run_name vqvae_standard_quantization \
                          --seed 1234 \  # fix seed for reproducibility
                          --logging \  # will log results to wandb
                          --workers 8               
```

### Evaluation

To evaluate a pre-trained model, run:
```
  python ./vqvae/evaluate.py --params_file "./example_confs/standard_vqvae_cb1024.yaml" \ # config of pretrained model
                             --dataloader ffcv \  # uses ffcv data-loader
                             --dataset_path "/home/datasets/examples/" \ # contains test.beton file
                             --batch_size 64 \ # evaluation is done on single gpu
                             --seed 1234 \  # fix seed for reproducibility
                             --loading_path "/home/runs/standard_vqvae_cb1024/last.ckpt" \ # checkpoint file
                             --workers 8             
```

The Evaluation process is based on the `torchmetrics` library (https://lightning.ai/docs/torchmetrics/stable/). For each run, 
computed measures are L2, PSNR, SSIM, rFID for reconstruction and Perplexity, Codebook usage on the whole test set for quantization.

###  Attempts to reproduce the original VQGAN results on _ImageNet-1K_

Reproduction is really hard, mainly due to the high compression rate (256x256 to 16x16) and relatively small
codebook size (1024 indices). 

The pretrained models and configuration files used can be downloaded at 
[this link](https://drive.google.com/drive/folders/1nUSYakY9R9DPxCNqjz26hSRa3bsFbvkJ?usp=sharing) 


| Run Name                      | Codebook Usage | Perplexity | L2     | SSIM | PSNR  | rFID | # (trainable) params |  
|-------------------------------|---------------:|-----------:|--------|------|-------|------|---------------------:|
| original VQGAN (Esser et Al.) |              - |          - | -      | -    | -     | 7.94 |                    - |
| Maskgit VQGAN  (Cheng et Al.) |              - |          - | -      | -    | -     | 2.28 |                    - |
| Gumbel Reproduction           |        99.61 % |     892.00 | 0.0075 | 0.61 | 21.23 | 6.30 |               72.5 M |


_Note:_ For training, NVIDIA A100 GPUs with Tensor Core have been used.

### Details on Quantization Algorithms

Classic or EMA VQ-VAE are known to encounter codebook-collapse issues, where only a subset of the codebook indices
is used. See for example: _Theory and Experiments on
Vector Quantized Autoencoders_ (https://arxiv.org/pdf/1805.11063.pdf)  

To avoid collapse, some solutions have been proposed (and are implemented in this repo):
1. Re-initialize the unused codebook indices every _n_ epochs. Can be applied with standard
or EMA Vector Quantization.
in the Gumbel Softmax and Entropy Quantization algorithms.
2. Totally change the Quantization algorithm, adding some regularization term (Gumbel, Entropy) to increase the entropy
in the codebook distribution.

### Details on Discriminator part 

In general, it is better to wait as long as possible before Discriminator kicks in.  
Check these issues in the original VQGAN repo:
- https://github.com/CompVis/taming-transformers/issues/31
- https://github.com/CompVis/taming-transformers/issues/61
- https://github.com/CompVis/taming-transformers/issues/93

In the reproduction, Discriminator starts only after 100 epochs. The training continues until possible. At a certain 
point, the loss collapses (typical behavior in GANs).  

I found that R1 regularization can help, while the adaptive generator weight does not improve results (used a fixed 0.1
weight on generator).
