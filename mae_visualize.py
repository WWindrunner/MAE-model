import sys
import os
#import requests
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('.')
import models_mae


def save_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.savefig("visualize/" + title)
    #pil_image = Image.fromarray(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    #pil_image.save("visualize/" + title)
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16', save_folder=""):
    # build model
    model = models_mae.__dict__[arch](img_size=336, save_path=save_folder)
    #model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, tag, file_name=""):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0, file_name=file_name)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    save_image(x[0], tag + "original.jpg")

    save_image(im_masked[0], tag + "masked.jpg")

    save_image(y[0], tag + "reconstruction.jpg")

    save_image(im_paste[0], tag + "reconstruction+visible.jpg")


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


chkpt_dir = '/tank/data/SFS/xinyis/shared/data/mae/output_dir_one_meter/336/checkpoint-399.pth'
save_path = "/tank/data/SFS/xinyis/shared/src/MUNIT/input_data/latent"
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16', save_path)
print('Model loaded.')

#img = Image.open("/tank/data/SFS/xinyis/shared/src/MUNIT/input_data/prepared/trainA/00000.jpg")
#img = img.resize((336, 336))
#img = np.array(img) / 255.
#assert img.shape == (336, 336, 3)
# normalize by ImageNet mean and std
#img = img - imagenet_mean
#img = img / imagenet_std
#print('Image loaded.')

# make random mask reproducible (comment out to make it change)
#torch.manual_seed(1)
#print('MAE with pixel reconstruction:')
#run_one_image(img, model_mae, "1", "00000")

# Batch generate latent code for MUNIT
input_folder = Path("/tank/data/SFS/xinyis/shared/src/MUNIT/input_data/prepared/trainA")
for f in input_folder.iterdir():
    if f.is_file() and f.suffix.lower() == ".jpg":
        print(f)
        with Image.open(f) as img:
            img = Image.open(f)
            img = img.resize((336, 336))
            img = np.array(img) / 255.
            img = img - imagenet_mean
            img = img / imagenet_std
            print('Image loaded.')
            print('MAE with pixel reconstruction:')
            run_one_image(img, model_mae, "1", f.stem)



