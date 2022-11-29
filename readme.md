# Vector Quantized Variational Auto-Encoder

pytorch (lightning) simple implementation for the vq-vae paper, using EMA codebook updating algorithm.

Original paper: https://arxiv.org/abs/1711.00937

Keras implementation: https://keras.io/examples/generative/vq_vae/

Original code by deepmind: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py

WandB logs are also included

### Training

1. Install requirements as in `requirements.txt`
2. your dataset can be composed of images in `.png .jpg .bmp .JPEG` formats.  
   The dataset structure must be like the following:
    ```
    :file_folder: path/to/dataset/
        :open_file_folder: train/
         ┣ 000.jpeg
         ┣ 001.jpg
         ┗ 002.png
        :open_file_folder: test/
         ┣ 003.jpeg
         ┣ 004.jpg
         ┗ 005.png
    ```
3. write a *.yaml* file containing model parameters. An example can be found in `/examples/params_file.yaml`
4. run training script like:  
  ```
  python3 train_vqvae.py --params_file './examples/params_file.yaml' 
                         --dataset_path '/path/to/dataset/' 
                         --checkpoint_path './runs/'
                         --run_name 'my-run-name'
                         --logging True
                         --seed 1234
                         --gpus 1
                         --workers 8
 ```