## Modified Masked Autoencoders for UWM HPC

This is the MAE model code I run on UWM HPC. I slightly modified the code to let it save the latent code after the encoder for my study purpose.


### Usage

#### Environment

To set up the conda environment, I have exported the environment settings in my home folder:

```bash
/home/uwm/maopuxu/mae_test.yml
```

Please install and activate miniconda, and then run the following commands:

```bash
conda env create -f /home/uwm/maopuxu/mae_test.yml
```

Hopefully the environment can be created successfully.


#### Training

`slurm_submit.sh` is the main entry point for trainning:

```python
python3 /home/uwm/maopuxu/MAE_Topography_Reconstruction/MAE-Topography/mae/main_pretrain.py --input_size 256 --data_path /tank/data/SFS/xinyis/shared/data/mae/one_meter_samples/OH_train_256 --output_dir /tank/data/SFS/xinyis/shared/data/mae/output_dir_one_meter/256 --log_dir /tank/data/SFS/xinyis/shared/data/mae/output_dir_one_meter/256
```

Here, replace `/home/uwm/maopuxu/MAE_Topography_Reconstruction/MAE-Topography/mae/main_pretrain.py` with path to your own `main_pretrain.py`.<br>
Replace `--input_size` with your desired input size (side length, or the number of pixels of the input image).<br>
Replace `--data_path` and `--output_dir` to your desired input and output location. You can take a look at the path above to check the input directory structure.<br>
`--log_dir` is just where to output the logs.<br>
More settings are available, and you can check the code for more details.

Run `sbatch slurm_submit.sh` to submit the job to HPC slurm system.


#### Testing

`mae_visualize.py` is the main entry point for testing/visualizing results.<br>
The structure is a bit messy since I haven't get time to organize it. Here is a breakdown of the contents:

```python
chkpt_dir = '/tank/data/SFS/xinyis/shared/data/mae/output_dir_one_meter/336/checkpoint-399.pth'
save_path = "/tank/data/SFS/xinyis/shared/src/MUNIT/input_data/latent"
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16', save_path)
print('Model loaded.')
```

This part loads in the trained model from path `chkpt_dir`. If you do not need to save the latent code, just skip this parameter and use `prepare_model(chkpt_dir, 'mae_vit_large_patch16')`

```python
run_one_image(img, model_mae, "1", f.stem)
```

This line processes one image `img` with the loaded `model_mae`. This "1" here is just any tag that you can name the image. The last input is the stored latent name, and you can skip it again if you do not need it.


### Model details

To change how you mask the images, take a look at line 128 in `models_mae.py`:

```python
def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
```

This function is the original masking method in MAE, and I wrote another `middle_masking` right after it.<br>
Please follow the same logic to write your own masking function, and replace that at line 213 in `models_mae.py`:

```python
# masking: length -> length * mask_ratio
x, mask, ids_restore = self.random_masking(x, mask_ratio)
```


Please contact **maopuxu@uwm.edu** for questions.