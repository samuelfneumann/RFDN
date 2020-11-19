# Import modules
import utils_image as util
from tqdm import tqdm
import pickle
from skimage.metrics import structural_similarity as ssim
import os
from skimage.io import imread
from skimage.transform import rescale


# Class definitions
class Interpolate:
    """
    Class Interpolate will interpolate images and calculate the PSNR and SSIM
    values of the interpolated images.
    """
    def __init__(self, data_dir):
        """
        Constructor

        Parameters
        ----------
        data_dir : str
            The absolute path to the data directory, which holds the Python
            dictionary of filenames for the validation data.

        Raises
        ------
        ValueError
            If the checkpoint file or data directory does not exist
        """
        # Get validation data filenames
        if not os.path.isdir(data_dir):
            raise ValueError("data_dir directory does not exist")

        filenames = "/valFilenames.bin"
        with open(data_dir + filenames, "rb") as data_file:
            self.val = pickle.load(data_file)

    def calculate_values(self, amount=2):
        """
        Calculates the PSNR and SSIM values of the interpolated images

        Parameters
        ----------
        amount : int, optional
            The amount to rescale the image by, by default 2

        Returns
        -------
        tuple of list of float
            A tuple of (psnr values, ssim values) for each image in the
            data directory.
        """
        psnr = []
        ssim_ = []
        for file in tqdm(list(self.val.keys())):
            img = imread(file)
            upscale = rescale(img, amount, multichannel=True)

            hr_img = imread(self.val[file])

            psnr.append(util.calculate_psnr(hr_img, upscale))
            ssim_.append(ssim(hr_img, upscale, multichannel=True))

        return psnr, ssim_
