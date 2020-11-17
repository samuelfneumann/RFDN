# Import modules
import torch
import os
import socket
import numpy as np
import matplotlib.pyplot as plt
import utils_image as util
import pickle
from pytorch_msssim import ssim
from tqdm import tqdm
from time import time

LOCAL_HOSTNAME = "alienware-15-r2"


# Class definitions
class Compare:
    """
    Class Compare compares two models in a few ways, which include plotting
    their learning curves as well as plotting side-by-side comparisons
    out predictions.
    """
    def __init__(self, model1, model2, checkpoint1, checkpoint2, data_dir):
        """
        Constructor, see class documentation for more details.

        Parameters
        ----------
        model1 : torch.nn.Module
            The first network to consider
        model2 : torch.nn.Module
            The second network to consider
        checkpoint1 : str
            The absolute path to the checkpoint for the first model
        checkpoint2 : str
            The absolute path to the checkpoint for the second model
        data_dir : str
            The absolute path to the data directory, which has the layout
            as specified in the readme file

        Raises
        ------
        ValueError
            If the checkpoint or data_dir files do not exist
        """
        self.model1 = model1
        self.model2 = model2

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        if self.device.type != "cuda":
            print("not using cuda")

        self.model1.to(self.device)
        self.model2.to(self.device)

        # Set both models to evaluation mode
        self.model1.eval()
        self.model2.eval()

        # Get validation data filenames
        if not os.path.isdir(data_dir):
            raise ValueError("data_dir directory does not exist")

        # Choose the appropriate file to use for the LR-HR data dictionary
        local = socket.gethostname().lower() == LOCAL_HOSTNAME
        filenames = "/valFilenames.bin" if local else "/valFilenamesDrive.bin"
        with open(data_dir + filenames, "rb") as data_file:
            self.val = pickle.load(data_file)

        # Load in model evaluations
        self.lc1 = self._load_checkpoint(self.model1, checkpoint1)
        self.lc2 = self._load_checkpoint(self.model2, checkpoint2)

    def plot_lc(self, type_="psnr", x=(None, None), y=(None, None)):
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
        type_ = type_.lower()
        if type_ not in ("psnr", "ssim", "loss"):
            raise ValueError("type_ must be one of 'psnr' or 'ssim' or 'loss'")

        # Create the appropriate figure and axis
        fig = plt.figure()
        ax = fig.add_subplot()

        # Plot the learning curves
        ax.plot(self.lc1[type_], label=type_ + " " + str(self.model1))
        ax.plot(self.lc2[type_], label=type_ + " " + str(self.model2))

        # Add labels and titles
        ax.set_ylabel(type_ + " " + str(self.model1))
        ax.set_xlabel("epochs")
        ax.set_title("Learning Curve (" + type_ + ") - " + str(self.model1))
        ax.legend()

        # Set the x/y limits
        if x != (None, None):
            ax.set_xlim(x[0], x[1])

        if y != (None, None):
            ax.set_ylim(y[0], y[1])

        fig.show()

    def predict(self, lr_img_name: str, img_name: str, save_img=True):
        """
        Produces the super-resolution of a single input image and outputs
        the PSNR and SSIM. Optionally saves the super-resoluted image in the
        current working directory.

        Parameters
        ----------
        lr_img_name : str
            The absolute path to the low resolution image.
        img_name : str
            The absolute path to the image to save
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
            prediction1 = self.model1(img_lr).cpu()
            prediction2 = self.model2(img_lr).cpu()

            # Generate performance measures on validation data for
            # learning curves
            psnr1 = util.calculate_psnr(prediction1.numpy(), img_hr.numpy())
            ssim_1 = ssim(prediction1, img_hr)
            psnr2 = util.calculate_psnr(prediction2.numpy(), img_hr.numpy())
            ssim_2 = ssim(prediction2, img_hr)

            print(f"PSNR for model 1 ({str(self.model1)}): {psnr1}")
            print(f"SSIM for model 1 ({str(self.model1)}): {ssim_1}")
            print(f"PSNR for model 2 ({str(self.model2)}): {psnr2}")
            print(f"SSIM for model 2 ({str(self.model2)}): {ssim_2}")

            avg_loss1 = np.mean(self.lc1["loss"])
            avg_loss2 = np.mean(self.lc2["loss"])
            print(f"Average Loss for model 1 ({str(self.model1)}):" +
                  f"{avg_loss1}")
            print(f"Average Loss for model 2 ({str(self.model2)}): " +
                  f"{avg_loss2}")

            # Save image if needed
            if save_img:
                print("Saving images")
                img1 = util.tensor2uint(prediction1)
                img2 = util.tensor2uint(prediction2)

                util.imsave(img1, img_name + str(self.model1) + "_1.jpg")
                util.imsave(img2, img_name + str(self.model2) + "_2.jpg")

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
        psnr = {"model1": [], "model2": []}
        ssim_ = {"model1": [], "model2": []}
        times = {"model1": [], "model2": []}
        with torch.no_grad():
            for lr_img_name in tqdm(list(self.val.keys())):
                # Open LR image
                img_lr = util.uint2tensor4(util.imread_uint(lr_img_name))
                img_lr = img_lr.to(self.device)

                # Open HR image label
                img_hr = util.uint2tensor4(util.imread_uint
                                           (self.val[lr_img_name]))
                img_hr = img_hr.cpu()

                # Time for prediction for model 1
                start_time = time()
                prediction1 = self.model1(img_lr).cpu()
                end_time = time()
                times["model1"].append(end_time - start_time)

                # Time for prediction for model 2
                start_time = time()
                prediction2 = self.model2(img_lr).cpu()
                end_time = time()
                times["model2"].append(end_time - start_time)

                # Generate performance measures on validation data
                psnr["model1"].append(util.calculate_psnr(prediction1.numpy(),
                                                          img_hr.numpy()))
                ssim_["model1"].append(ssim(prediction1, img_hr))
                psnr["model2"].append(util.calculate_psnr(prediction2.numpy(),
                                                          img_hr.numpy()))
                ssim_["model2"].append(ssim(prediction2, img_hr))

        # Save the performance evaluation measures to the Trainer
        return {"psnr": psnr, "ssim": ssim_, "times": times}

    def compare_patches(self, index: int, size=24, start=(0, 0),
                        figsize=(15, 24)):
        """
        Compares patches of predictions for the two models

        Parameters
        ----------
        index : int
            The index of the LR file in the validation file dictionary keys
        size : int, optional
            The size of the patch, by default 24
        start : tuple, optional
            The coordinate of the first pixel in the uper-resoluted patch,
            which is the top left pixel in the patch, by default (0, 0)
        figsize : tuple, optional
            The size of the matplotlib figure, by default (15, 24)
        """
        # Load in the low resolution input
        lr_img_file = list(self.val.keys())[index]
        lr_img = util.uint2tensor4(util.imread_uint(lr_img_file))
        lr_img = lr_img.to(self.device)

        with torch.no_grad():
            # Get the first model's prediction
            prediction1 = self.model1(lr_img).cpu().squeeze(0).clamp_(0, 255)
            prediction1 = prediction1.permute(1, 2, 0).numpy() / 255

            # Get the second model's prediction
            prediction2 = self.model2(lr_img).cpu().squeeze(0).clamp_(0, 255)
            prediction2 = prediction2.permute(1, 2, 0).numpy() / 255

        # Separate the two predictions into patches
        prediction1_patch = prediction1[start[0]:start[0] + size,
                                        start[1]:start[1]+size]
        prediction2_patch = prediction2[start[0]:start[0] + size,
                                        start[1]:start[1]+size]

        # Create the relevant figure and axes
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Plot the first prediction patch
        ax1.imshow(prediction1_patch)
        ax1.set_title("  (a) - model " + str(self.model1), y=0.01, loc="left",
                      color="white", fontsize=30)
        ax1.set_axis_off()

        # Plot the second prediction patch
        ax2.imshow(prediction2_patch)
        ax2.set_title("  (b) - model " + str(self.model2), y=0.01, loc="left",
                      color="white", fontsize=30)
        ax2.set_axis_off()

        fig.tight_layout()

    def _load_checkpoint(self, model, checkpoint):
        """
        Loads in a checkpoint file, returning the learning curves

        Parameters
        ----------
        model : torch.nn.Module
            One of the two models that are being compared
        checkpoint : str
            The absolute path to the checkpoint file

        Returns
        -------
        dict
            The learning curves dictionary

        Raises
        ------
        ValueError
            If the checkpoint file does not exist
        """
        # Ensure checkpoint file exists
        if not os.path.isfile(checkpoint):
            raise ValueError("checkpoint file does not exist")

        # Open and load the checkpoint
        checkpoint = torch.load(checkpoint)

        lc = checkpoint["lc"]
        model.load_state_dict(checkpoint["model_param"])

        return lc
