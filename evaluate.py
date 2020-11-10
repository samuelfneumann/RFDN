# Import modules
import numpy as np
import pickle
import torch
import gc
import os
import matplotlib.pyplot as plt
from time import time
from utils import utils_image as util
from pytorch_msssim import ssim
from random import shuffle
from tqdm import tqdm


class Evaluate():
    """
    Class Evaluate takes a model and performs evaluation metrics on it through
    the validation data as well as the information saved in the training
    process through checkpointing.
    """
    def __init__(self, model, checkpoint_file, data_dir):
        """
        Constructor

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate
        checkpoint_file : str
            The absolute path to the checkpoint file created using a
            train.Trainer object.
        data_dir : str
            The absolute path to the data directory, which holds the Python
            dictionary of filenames for the validation data.

        Raises
        ------
        ValueError
            If the checkpoint file or data directory does not exist
        """
        # Choose appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        if self.device.type != "cuda":
            print("not using cuda")
        # torch.cuda.current_device()
        torch.cuda.empty_cache()

        # Get validation data filenames
        if not os.path.isdir(data_dir):
            raise ValueError("data_dir directory does not exist")
        with open(data_dir + "/valFilenames.bin", "rb") as data_file:
            self.val = pickle.load(data_file)

        # Ensure checkpoint file exists
        if not os.path.isfile(checkpoint_file):
            raise ValueError("checkpoint file does not exist")

        # Open and load the checkpoint
        checkpoint = torch.load(checkpoint_file)

        self.lc = checkpoint["lc"]
        self.epoch = checkpoint["epoch"]
        self.model = model
        self.model.eval()
        self.model.load_state_dict(checkpoint["model_param"])
        self.model.to(self.device)

    def plot_lc(self, type_="psnr"):
        """
        Plots the learning curves from the data generated when training.

        Parameters
        ----------
        type_ : str, optional
            The type of learning curve to plot, which must be one of "psnr",
            "ssim", or "loss", by default "psnr"

        Raises
        ------
        ValueError
            If type_ is not one of the allowable types.
        """
        if (type_ := type_.lower()) not in ("psnr", "ssim", "loss"):
            raise ValueError("type_ must be one of 'psnr' or 'ssim' or 'loss'")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.lc[type_], label=type_)

        ax.set_ylabel(type_)
        ax.set_xlabel("epochs")
        ax.set_title("Learning Curve")

        ax.legend()

        fig.show()

    def predict(self, lr_img_name: str, save_img=True):
        """
        Produces the super-resolution of a single input image and outputs
        the PSNR and SSIM. Optionally saves the super-resoluted image in the
        current working directory.

        Parameters
        ----------
        lr_img_name : str
            The absolute path to the low resolution image.
        save_img : bool, optional
            Whether the image should be save or not, by default True
        """
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

            # Save image if needed
            if save_img:
                img = util.tensor2uint(prediction)
                util.imsave(img, "./img.jpg")

    def get_values(self):
        """
        Produces the values of the PSNR and SSIM for each validation data
        instance, as well as the the time to upscale each image.

        Returns
        -------
        dict of list of float
            A dictionary containing the lists of PSNR, SSIM, and inference time
            values for each validation data instance.
        """
        psnr = []
        ssim_ = []
        times = []
        with torch.no_grad():
            for lr_img_name in tqdm(list(self.val.keys())):
                # Open LR image
                img_lr = util.uint2tensor4(util.imread_uint(lr_img_name))
                img_lr = img_lr.to(self.device)

                # Open HR image label
                img_hr = util.uint2tensor4(util.imread_uint
                                           (self.val[lr_img_name]))
                img_hr = img_hr.cpu()

                # Time and predict
                start_time = time()
                prediction = self.model(img_lr).cpu()
                end_time = time()
                times.append(end_time - start_time)

                # Generate performance measures on validation data for
                # learning curves
                psnr.append(util.calculate_psnr(prediction.numpy(),
                                                img_hr.numpy()))
                ssim_.append(ssim(prediction, img_hr))

        # Save the performance evaluation measures to the Trainer
        return {"psnr": psnr, "ssim": ssim_, "times": times}