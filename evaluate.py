# Import modules
import numpy as np
import pickle
import torch
import gc
import os
import matplotlib.pyplot as plt
from utils import utils_image as util
from pytorch_msssim import ssim
from random import shuffle
from tqdm import tqdm
from RFDN import RFDN


class Evaluate():
    def __init__(self, model, checkpoint_file, data_dir):
        # Choose appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            print("not using cuda")
        # torch.cuda.current_device()
        torch.cuda.empty_cache()

        # Get validation data filenames
        with open(data_dir + "/valFilenames.bin", "rb") as data_file:
            self.val = pickle.load(data_file)

        # Ensure checkpoint file exists
        if not os.path.isfile(checkpoint_file):
            raise ValueError("checkpoint file does not exist")

        # Open and load the checkpoint
        # with open(checkpoint_file, "rb") as infile:
        #     checkpoint = pickle.load(infile)

        checkpoint = torch.load(checkpoint_file)
        self.lc = checkpoint["lc"]
        self.epoch = checkpoint["epoch"]
        self.model = model
        self.model.eval()
        self.model.load_state_dict(checkpoint["model_param"])
        self.model.to(self.device)
        # self.model.load_state_dict(checkpoint["parameters"])

    def plot_lc(self, type_="psnr"):
        if (type_ := type_.lower()) not in ("psnr", "ssim"):
            raise ValueError("type_ must be one of 'psnr' or 'ssim'")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.lc[type_])

        ax.set_xlabel("epochs")
        ax.set_ylabel(type_)

        fig.show()

    def predict(self, lr_img_name):
        with torch.no_grad():
            img_lr = util.uint2tensor4(util.imread_uint(lr_img_name))
            img_lr = img_lr.to(self.device)

            # Open HR image label
            img_hr = util.uint2tensor4(util.imread_uint(self.val[lr_img_name]))
            img_hr = img_hr.cpu()

            # Predict
            prediction = self.model(img_lr).cpu()

            # Generate performance measures on validation data for
            # learning curves
            psnr = util.calculate_psnr(prediction.numpy(), img_hr.numpy())
            ssim_ = ssim(prediction, img_hr)

            print(f"PSNR: {psnr}")
            print(f"SSIM: {ssim_}")

            img = util.tensor2uint(prediction)
            util.imsave(img, "./img.jpg")

