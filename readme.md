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

Discriminator cloned from:  https://github.com/NVlabs/stylegan2-ada-pytorch  
Discriminator Losses (hinge / non-saturating): https://github.com/google-research/maskgit

Quantization Algorithms: 
   - Standard and EMA update: Original VQVAE paper. 
   - Gumbel Softmax: https://github.com/karpathy/deep-vector-quantization
   - "Entropy" Quantizer: https://github.com/google-research/maskgit

Fast Data Loading:
   - FFCV: https://github.com/libffcv/ffcv
   - FFCV_PL: https://github.com/SerezD/ffcv_pytorch_lightning

### Details on Quantization Algorithms

A known problem of VQ-VAE is codebook-collapse, where only a subset of the codebook indices
is used.  
See for example: _Theory and Experiments on
Vector Quantized Autoencoders_ (https://arxiv.org/pdf/1805.11063.pdf)  

To avoid collapse, some solutions have been proposed (and are implemented in this repo):
1. Re-initialize the unused codebook indices every _n_ epochs. Can be applied with standard
or EMA Vector Quantization.
2. Add a KL loss term to regularize the codebook distribution to a uniform prior. This is used
in the Gumbel Softmax and Entropy Quantization algorithms.

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
     📂 test/
      ┣ 006.jpeg
      ┣ 007.jpg
      ┗ 008.png
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
   - `****_vqvae_cb1024`: train a base vqvae with different quantization algorithms.
   - `standard_vqvae_cb4096` and `standard_vqvae_cb4096_reinit10like`: test how reinitializing unused codes improves codebook usage.
   - `standard_vqgan_cb1024`: train a vqgan model with StyleGan discriminator, using adaptive generator weight and R1 regularization.

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


###  Pretrained Models, Configuration Files and Training Logs

(WORK IN PROGRESS)

The Evaluation process uses torchmetrics to compute L2, PSNR, SSIM, rFID
Plus perplexity and codebook usage on the test set. 

RUN NAME codebook usage, perplexity, L2, SSIM, PSNR, rFID, nGPUS * Nodes, gpu/hours

VQVAE CODEBOOK 1024 STANDARD QUANTIZATION
|             mse            │   0.004431578796356916    │
│        perplexity         │      733.31982421875      │
│           psnr            │    23.534406661987305     │
│           rfid            │          52.0625          │
│           ssim            │    0.6876464486122131     │
│       used_codebook       │        99.70703125
