# Import modules
import numpy as np
import skimage.transform
import skimage.io
import pickle
import torch
import socket
from train import LOCAL_HOSTNAME
# import gc
import os
import matplotlib.pyplot as plt
from time import time
import utils_image as util
from pytorch_msssim import ssim
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

        # Choose the appropriate file to use for the LR-HR data dictionary
        local = socket.gethostname().lower() == LOCAL_HOSTNAME
        filenames = "/valFilenames.bin" if local else "/valFilenamesDrive.bin"
        with open(data_dir + filenames, "rb") as data_file:
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
        type_ = type_.lower()
        if type_ not in ("psnr", "ssim", "loss"):
            raise ValueError("type_ must be one of 'psnr' or 'ssim' or 'loss'")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.lc[type_], label=type_)

        ax.set_ylabel(type_)
        ax.set_xlabel("epochs")
        ax.set_title("Learning Curve (" + type_ + ")")

        ax.legend()

        fig.show()

    def predict(self, lr_img_num: int, img_name: str, save_img=True):
        """
        Produces the super-resolution of a single input image and outputs
        the PSNR and SSIM. Optionally saves the super-resoluted image in the
        current working directory.

        Parameters
        ----------
        lr_img_num : int
            The index of the low resolution image in the validation data
            dictionary keys.
        img_name : str
            The absolute path to the image to save
        save_img : bool, optional
            Whether the image should be save or not, by default True
        """
        lr_img_name = list(self.val.keys())[lr_img_num]
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
            avg_loss = np.mean(self.lc["loss"])
            print(f"Average Loss: {avg_loss}")

            # Save image if needed
            if save_img:
                print("Saving image")
                img = util.tensor2uint(prediction)

                util.imsave(img, img_name)

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
            The coordinate of the first pixel in the super-resoluted patch,
            which is the top left pixel in the patch, by default (0, 0)
        figsize : tuple, optional
            The size of the matplotlib figure, by default (15, 24)
        """
        # Load in the LR input
        lr_img_file = list(self.val.keys())[index]
        lr_img = util.uint2tensor4(util.imread_uint(lr_img_file))
        lr_img = lr_img.to(self.device)

        # Get the network's prediction
        with torch.no_grad():
            prediction = self.model(lr_img).cpu().squeeze(0).clamp_(0, 255)
            prediction = prediction.permute(1, 2, 0).numpy() / 255

        # Get the HR label
        hr_img_file = self.val[lr_img_file]
        hr_img = util.imread_uint(hr_img_file) / 255

        # Separate the label and prediction into patches
        hr_patch = hr_img[start[0]:start[0] + size, start[1]:start[1]+size]
        prediction_patch = prediction[start[0]:start[0] + size,
                                      start[1]:start[1]+size]

        # Set up the figure to plot
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Plot the prediction
        ax1.imshow(prediction_patch)
        ax1.set_title("  (a) - prediction", y=0.01, loc="left", color="white",
                      fontsize=30)
        ax1.set_axis_off()

        # Plot the HR label
        ax2.imshow(hr_patch)
        ax2.set_title("  (b) - HR label", y=0.01, loc="left", color="white",
                      fontsize=30)
        ax2.set_axis_off()

        fig.tight_layout()

    def compare_interpolation(self, index: int, size=24, start=(0, 0),
                              figsize=(15, 24)):
        # Load in the LR input
        lr_img_file = list(self.val.keys())[index]
        lr_img = util.uint2tensor4(util.imread_uint(lr_img_file))
        lr_img = lr_img.to(self.device)

        # Get the network's prediction
        with torch.no_grad():
            prediction = self.model(lr_img).cpu().squeeze(0).clamp_(0, 255)
            prediction = prediction.permute(1, 2, 0).numpy() / 255

        # Read in the image and upsample it through interpolation
        interpolated = skimage.io.imread(lr_img_file)
        interpolated = skimage.transform.rescale(interpolated, 2,
                                                 multichannel=True)

        # Separate the label and prediction into patches
        interp_patch = interpolated[start[0]:start[0] + size,
                                    start[1]:start[1]+size]
        prediction_patch = prediction[start[0]:start[0] + size,
                                      start[1]:start[1]+size]

        # Set up the figure to plot
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Plot the prediction
        ax1.imshow(prediction_patch)
        ax1.set_title("  (a) - prediction", y=0.01, loc="left", color="white",
                      fontsize=30)
        ax1.set_axis_off()

        # Plot the HR label
        ax2.imshow(interp_patch)
        ax2.set_title("  (b) - interpolation", y=0.01, loc="left",
                      color="white", fontsize=30)
        ax2.set_axis_off()

        fig.tight_layout()


# if __name__ == "__main__":
#     # model = RFDN1(nf=10, upscale=2)
    # checkpoint_file = "/home/samuel/Documents/CMPUT511/Project/" + \
    #     "Checkpoints/checkpoint_3.tar"
#     data_dir = "/home/samuel/Documents/CMPUT511/Project/Data"
#     e = Evaluate(model, checkpoint_file, data_dir)

    # e.compare_prediction("/home/samuel/Documents/CMPUT511/Project" + \
    #     "/Data/val/LR/DIV2K_valid_LR_bicubic/X2/0801x2.png")
