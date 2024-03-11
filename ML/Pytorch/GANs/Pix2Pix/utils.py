import os

import numpy as np
import torch
import config
from torchvision.utils import save_image
from PIL import Image

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        input_image = x * 0.5 + 0.5  # input image after removing normalization
        # save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        combined_image = torch.cat([input_image, y * 0.5 + 0.5, y_fake], dim=2)
        # Save the combined image
        save_image(combined_image, folder + f"/combined_{epoch}.png")
        # if epoch == 1:
        #     save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(input_folder, output_folder, gen):
    files = os.listdir(input_folder)

    gen.eval()
    for file in files:
        image = Image.open(os.path.join(input_folder, file))
        with torch.no_grad():
            line_img = gen(
                config.transform_only_input(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        save_image(line_img * 0.5 + 0.5, os.path.join(output_folder, file))
    gen.train()