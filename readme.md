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
   - Gumbel Softmax: code taken from https://github.com/karpathy/deep-vector-quantization, 
     parameters from DALL-E paper: https://arxiv.org/abs/2102.12092. Also check: https://arxiv.org/abs/1611.01144
     for theory.
   - "Entropy" Quantizer: code taken from https://github.com/google-research/maskgit

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
you want to train.

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

###  Pretrained Models, Configuration Files and Training Logs

In this section, you can find the evaluation results obtained for some _ImageNet-1K_ pretrained models.

The pretrained models, training log and configuration files used can be downloaded at 
[this link](https://drive.google.com/drive/folders/1nUSYakY9R9DPxCNqjz26hSRa3bsFbvkJ?usp=sharing) 

List of tested models and short description:

**Ablations: different quantization algorithms**
- **standard_vqvae_cb4096**: replication of the base model, as described in the original vqvae paper.
Uses a large codebook with 4096 entries and no reinitialization. Enlights the problem of codebook collapse 
(only half of the codes are used at inference time).
- **standard_vqvae_cb4096_reinit10like**: same run as previous, unused codes are re-initialized every 10 epochs. 
Shows how reinitialization can help in preventing codebook collapse, also with slightly better reconstruction results.
- **ema_vqvae_cb4096**: Standard VAE using the EMA (**exponential moving average**) algorithm for learning the codebook, as 
described in the Appendix of VQVAE paper. Results are similar to the base run, with a small reduction in trainable 
parameters (since codebook is not directly trained).
- **entropy_vqvae_cb4096**: Test the quantization algorithm showed in MaskGit code. 
I could not find any references in the code or the paper, so I don't exactly know where this comes from. 
Anyway, codebook usage is not high, but the codes are very well distributed (high perplexity). 
Also, rFID seems positively affected.
- **gumbel_vqvae_cb4096**: Uses gumbel-softmax trick during quantization. This has also been used in the DALL-E paper. 
It achieves the better codebook usage, but reconstructions metrics are the worst. Note that his may be due to a poor choice
of hyperparameters.

| Run Name                            | Codebook Usage |    Perplexity | L2         | SSIM      | PSNR      | rFID      |    N gpus * hours / epochs |    # (trainable) params |  
|-------------------------------------|---------------:|--------------:|------------|-----------|-----------|-----------|---------------------------:|------------------------:|
| standard_vqvae_cb4096               |        47.14 % |       1328.33 | 0.0042     | 0.69      | 23.77     | 50.12     |                  **2.105** |                  36.6 M |
| standard_vqvae_cb4096_reinit10like  |        88.45 % |       2538.91 | **0.0039** | **0.70**  | **24.06** | **47.06** |                      2.110 |                  36.6 M |
| ema_vqvae_cb4096                    |        75.09 % |       1693.17 | 0.0040     | **0.70**  | 23.97     | 48.06     |                      2.118 |              **35.6 M** |
| entropy_vqvae_cb4096                |        64.16 % |       2374.97 | 0.0044     | 0.68      | 23.50     | 47.50     |                      2.194 |                  36.6 M |
| gumbel_vqvae_cb4096                 |    **96.99 %** |   **3475.23** | 0.0046     | 0.67      | 23.38     | 59.25     |                      2.358 |                  55.4 M |

**Ablations: test the GAN Loss part**

- **standard_vqvae_cb1024**: replication of the base model, as described in the original vqvae paper. 
Uses a limited codebook of only 1024 entries, serving as baseline with later tests. On such a complex dataset as Imagenet-1K, 
performance is poor.
- **standard_vqgan_cb1024_noDisc**: Reconstruction Loss is a combination of L2, L1 and Perceptual. 
No Discriminator is used. The result is an improvement of the rFID metric w.r.t the base vqvae. 
All other metrics are performing worse.
- **standard_vqgan_cb1024_fixed**: Reconstruction Loss is a combination of L2, L1, Perceptual and Adversarial.
StyleGan Discriminator is used, increasing training time and parameters. Also note that the learning rate is now 1e-4. 
I tested with 3e-4 as in the previous runs, but NAN values in the generator/discriminator loss appeared. 
The generator loss has a fixed weight applied. rFID still improves, while all other metrics are getting worse w.r.t the vqvae baseline.
- **standard_vqgan_cb1024_adaptive**: Same run as the previous, but with adaptive weight applied on the generator loss, 
as described in the taming transformer paper. Training time increases due to the gradient calculation of the adaptive weight.
Apparently, the adaptive weight is worth since it improves all the reconstruction metrics w.r.t. the "fixed" case. 
Interestingly, only 91% of the codebook is used at inference time.
- **standard_vqgan_cb1024_fixed_R1**: Extends the **standard_vqgan_cb1024_fixed** run. Uses a fixed weight for the generator loss and R1 regularization 
on Discriminator loss. R1 helps in obtaining better metric everywhere, while slightly increasing training time.
- **standard_vqgan_cb1024_adaptive_R1**: Extends the **standard_vqgan_cb1024_adaptive** run. All implemented tricks are used 
for this run for the reconstruction loss part. Has the adaptive weight for generator loss and R1 regularization for Stylegan-Discriminator loss. 
Again, R1 helps in obtaining slightly better results w.r.t. the baseline run (**standard_vqgan_cb1024_adaptive**).


| Run Name                           |    Codebook Usage |    Perplexity | L2          | SSIM      | PSNR      | rFID       | N gpus * hours / epochs |     # (trainable) params |  
|------------------------------------|------------------:|--------------:|-------------|-----------|-----------|------------|------------------------:|-------------------------:|
| standard_vqvae_cb1024              |           99.71 % |        733.32 | **0.0044**  | **0.69**  | **23.53** | 52.06      |               **2.108** |               **35.8 M** |
| standard_vqgan_cb1024_noDisc       |           99.71 % |    **754.93** | 0.0047      | 0.67      | 23.28     | 30.84      |                   2.200 |               **35.8 M** |
| standard_vqgan_cb1024_fixed        |           99.71 % |        738.57 | 0.0068      | 0.60      | 21.68     | 28.87      |                   8.002 |                   64.7 M |
| standard_vqgan_cb1024_adaptive     |           91.02 % |        702.22 | 0.0054      | 0.65      | 22.67     | 21.56      |                   9.121 |                   64.7 M |
| standard_vqgan_cb1024_fixed_R1     |      **100.00 %** |        737.21 | 0.0067      | 0.61      | 21.71     | 25.98      |                   8.952 |                   64.7 M |
| standard_vqgan_cb1024_adaptive_R1  |           92.67 % |        720.19 | 0.0053      | 0.66      | 22.75     | **20.95**  |                  10.223 |                   64.7 M |


**Test different quantization algorithms with VQGAN**

- **standard_vqgan_cb1024_adaptive_R1**: Same run as above, reported here for better comparison.
- **ema_vqgan_cb1024_adaptive_R1**: Despite a slightly better codebook usage, all other metrics perform equal or worse 
w.r.t. the standard algorithm (including perplexity).
- **entropy_vqgan_cb1024_adaptive_R1**: Despite higher perplexity, the percentage of codebook usage is low, probably
causing the bad performance in reconstruction metrics.
- **gumbel_vqgan_cb1024_adaptive_R1**: obtains a perfect codebook usage and very high perplexity, but cannot beat the
standard run on rFID.

| Run Name                          |    Codebook Usage |    Perplexity | L2          | SSIM       | PSNR       | rFID        | N gpus * hours / epochs | # (trainable) params |  
|-----------------------------------|------------------:|--------------:|-------------|------------|------------|-------------|------------------------:|---------------------:|
| standard_vqgan_cb1024_adaptive_R1 |           92.67 % |        720.19 | **0.0053**  | **0.66**   | 22.75      | **20.95**   |                  10.223 |               64.7 M |
| ema_vqgan_cb1024_adaptive_R1      |           99.51 % |        677.71 |  **0.0053** | **0.66**   | 22.74      | 21.20       |              **10.082** |           **64.4 M** |
| entropy_vqgan_cb1024_adaptive_R1  |           80.85 % |        788.82 | 0.0056      | 0.64       | 22.52      | 26.20       |                  10.107 |               64.7 M |
| gumbel_vqgan_cb1024_adaptive_R1   |      **100.00 %** |    **957.94** | **0.0053**  | 0.64       | **22.79**  | 23.05       |                  10.335 |               66.1 M |

_Note:_ For training, NVIDIA A100 GPUs with Tensor Core have been used.
