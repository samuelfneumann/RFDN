# Import Modules
import numpy as np
import pickle
import torch
import gc
import os
import utils_image as util
from pytorch_msssim import ssim
from random import shuffle
from tqdm import tqdm
from RFDN import RFDN


class Trainer():
    ITEMS_PER_CALCULATION = 75
    """
    Class Trainer trains a neural network on a super-resolution task. It
    tracks and checkpoints the process and also calculates validation data
    for learning curves.
    """
    def __init__(self, model, checkpoint_file, data_dir, lc=True):
        """
        Constructor

        Parameters
        ----------
        model : torch.nn.Module
            The network to train
        checkpoint_file : str
            The desired filename for the file to keep the checkpoint of. This
            should be an absolute path.
        data_dir : str
            The absolute path to the directory which contains the training and
            validation data.
        lc : bool
            Whether or not to save the learning curve data, by default True.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        if self.device.type != "cuda":
            print("not using cuda")
        # torch.cuda.current_device()
        torch.cuda.empty_cache()

        self.model = model
        self.model = self.model.to(self.device)

        self.checkpoint_file = checkpoint_file
        self.epoch = 0

        # Get training data filenames
        self.data = None
        with open(data_dir + "/dataFilenamesDrive.bin", "rb") as data_file:
            self.data = pickle.load(data_file)

        # Get validation data filenames
        with open(data_dir + "/valFilenamesDrive.bin", "rb") as data_file:
            self.val = pickle.load(data_file)

        # Set up optimizer and loss
        self.lr = 1e-2
        self.optim = torch.optim.Adam(params=model.parameters(), lr=self.lr)
        self.criterion = torch.nn.L1Loss(reduction="mean")

        # Store PSNR for learning curves
        self.psnr_values = []
        self.ssim_values = []
        self.loss_values = []
        self.store_learning_curves = lc

    def load(self):
        """
        Loads the data in the checkpoint file into the current model and
        Trainer.
        """
        # Check checkpoint data
        checkpoint = torch.load(self.checkpoint_file)
        # with open(self.checkpoint_file, "rb") as infile:
        #     checkpoint = pickle.load(infile)

        # Load checkpoint data in
        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_param"])
        self.optim.load_state_dict(checkpoint["optim_param"])
        self.lr = checkpoint["lr"]
        for param in self.optim.param_groups:
            param["lr"] = self.lr

        # Keep learning curves data
        lc = checkpoint["lc"]
        self.psnr_values = lc["psnr"]
        self.ssim_values = lc["ssim"]
        self.loss_values = lc["loss"]
        print(f"Loading checkpoint on epoch {self.epoch}")

    def save(self, losses_per_epoch):
        """
        Saves a checkpoint of the model's parameters during training as well
        as the performance evaluation for the learning curves.

        Parameters
        ----------
        device : torch.device
            The device used to make the model's predictions
        losses_per_epoch : iter of float
            The losses per training example during training for each epoch
        """
        # Create the learning curves data
        lc = self.generate_learning_curves(losses_per_epoch)

        # Create the checkpoint data
        checkpoint = {"epoch": self.epoch,
                      "lr": self.lr,
                      "model_param": self.model.state_dict(),
                      "optim_param": self.optim.state_dict(),
                      "lc": lc}

        # Save the checkpoint data
        checkpoint_name = os.path.dirname(self.checkpoint_file) + \
            "/checkpoint_" + str(self.epoch) + ".tar"
        torch.save(checkpoint, checkpoint_name)
        # with open(self.checkpoint_file, "wb") as outfile:
        #     pickle.dump(checkpoint, outfile)

        # Save the learning curve data
        # if self.store_learning_curves:
        #     self.save_learning_curves(device, losses_per_epoch)

    def _save_lc_values(self):
        """
        Saves the data values for the learning curves

        Parameters
        ----------
        device : torch.device
            The device used to make the model's predictions
        """
        psnr = []
        ssim_ = []
        self.model.eval()
        with torch.no_grad():
            for lr_img_name in np.random.choice(list(self.val.keys()),
                                                Trainer.ITEMS_PER_CALCULATION):
                # Open LR image
                img_lr = util.uint2tensor4(util.imread_uint(lr_img_name))
                img_lr = img_lr.to(self.device)

                # Open HR image label
                img_hr = util.uint2tensor4(util.imread_uint
                                           (self.val[lr_img_name]))
                img_hr = img_hr.cpu()

                # Predict
                prediction = self.model(img_lr).cpu()

                # Generate performance measures on validation data for
                # learning curves
                psnr.append(util.calculate_psnr(prediction.numpy(),
                                                img_hr.numpy()))
                ssim_.append(ssim(prediction, img_hr))

        # Save the performance evaluation measures to the Trainer
        self.psnr_values.append(np.mean(psnr))
        self.ssim_values.append(np.mean(ssim_))

        # Return the model to training mode
        self.model.train()

    def generate_learning_curves(self, losses_per_epoch):
        """
        Generates the learning curves data. This includes all the
        performance measures on the validation data up to this point as well
        as the training losses per epoch.

        Parameters
        ----------
        device : torch.device
            The device used to make the model's predictions
        losses_per_epoch : iter of float
            The losses per training instance for each epoch while training

        Returns
        -------
        dict
            The learning curve data
        """
        # Save validation data for learning curves - PSNR and SSIM
        self._save_lc_values()

        # Save training loss for learning curves
        self.loss_values.append(np.mean(losses_per_epoch))

        # Save learning curve data
        values = {}
        values["psnr"] = self.psnr_values
        values["ssim"] = self.ssim_values
        values["loss"] = self.loss_values
        values["val_items"] = Trainer.ITEMS_PER_CALCULATION
        # with open("lc.bin", "wb") as lc_file:
        #     pickle.dump(values, lc_file)
        return values

    def train(self, num_epochs: int, load=False):
        """
        Trains the model, ensuring that the model is checkpointed after
        each epoch.

        Parameters
        ----------
        num_epochs : int
            The number of epochs to train
        load : bool, optional
            Whether to use the last checkpoint to load in model parameters and
            learning curves data or not, by default False
        """
        # Load the model if desirable
        if load:
            self.load()

        # Initialize model for training
        # self.model = self.model.to(self.device)
        self.model.train()

        # Train
        for _ in range(num_epochs):  # epochs
            losses_per_epoch = []
            lr_training_data = list(self.data.keys())
            shuffle(lr_training_data)
            for lr_img_name in tqdm(lr_training_data):
                # Open LR image
                img_lr = util.uint2tensor4(util.imread_uint(lr_img_name))
                img_lr = img_lr.to(self.device)

                # Open HR image
                img_hr = util.uint2tensor4(util.imread_uint
                                           (self.data[lr_img_name]))
                img_hr = img_hr.to(self.device)

                # Inference
                prediction = self.model(img_lr)

                # Calculate loss
                loss = self.criterion(prediction, img_hr)
                losses_per_epoch.append(float(loss))

                # Delete un-needed values
                del prediction
                del img_lr
                del img_hr
                torch.cuda.empty_cache()
                gc.collect()

                # Update parameters
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # Update epoch number and learning rate
            self.epoch += 1
            self.lr /= 2
            for param in self.optim.param_groups:
                param["lr"] = self.lr

            # Checkpoint
            self.save(losses_per_epoch)


# Train the model
if __name__ == "__main__":
    model = RFDN(nf=5, upscale=2)
    data_dir = "/home/samuel/Documents/CMPUT511/Project/Data"
    checkpoint_file = "/home/samuel/Documents/CMPUT511/Project/" + \
                      "Checkpoints/checkpoint.bin"
    trainer = Trainer(model, checkpoint_file, data_dir)
    trainer.train(1)
