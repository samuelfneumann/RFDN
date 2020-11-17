# Import modules
import numpy as np
import matplotlib.pyplot as plt
import os
from torch import load
# from scipy.interpolate import make_interp_spline, BSpline


# CLass definition
class LearningCurve:
    """
    Class LearningCurve will plot the average learning curve for a number
    of trained networks using the checkpoint files generated by the
    train.Trainer class.
    """
    def __init__(self, checkpoint_dir, **kwargs):
        """
        Constructor, see class documentation for more details.

        Parameters
        ----------
        checkpoint_dir : str
            The absolute path to the directory containing the checkpoint files
        """
        # Get checkpoint filesnames
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_files = os.listdir(checkpoint_dir)

        self.checkpoint_files2 = None
        self.checkpoint_dir2 = None
        if "checkpoint_dir2" in kwargs.keys():
            self.checkpoint_dir2 = kwargs["checkpoint_dir2"]
            self.checkpoint_files2 = os.listdir(self.checkpoint_dir2)

    def plot(self, type_="psnr", figsize=(16, 9), x=(None, None),
             y=(None, None), confidence=0.99, **kwargs):
        """
        Plots the learning curves for the model using the average learning
        values found in the checkpoint files. Plots the learning curves for
        the metric defined by type_.

        Parameters
        ----------
        type_ : str, optional
            The type of learning curve to plot, which must be one of 'psnr',
            'ssim', or 'loss', by default "psnr"
        figsize : tuple, optional
            The size of the figure, by default (16, 9)
        x : tuple of float, float
            The min/max x values for the x-axis
        y : tuple of float, float
            The min/max y values for the y-axis
        confidence : float
            The confidence level for the confidence interval about the
            learning curve.

        Returns
        -------
        np.array of np.float64
            The average values for the learning curve, which are then plotted

        Raises
        ------
        ValueError
            If the type_ argument is not one of 'psnr', 'ssim' or 'loss'
            If the confidence value is less than zero
        """
        if type_ not in ("psnr", "ssim", "loss"):
            raise ValueError("error: type_ must be one of 'psnr', 'ssim' or " +
                             "'loss'")

        if confidence < 0:
            raise ValueError(f"error: cannot construct confidence interval" +
                             f"with negative confidence value {confidence}")

        # Create the figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        # Calculate the average type_ values for the learning curve and plot
        label = ""
        means, errors = self._calculate_values(self.checkpoint_files,
                                               self.checkpoint_dir, type_)
        if "label" in kwargs.keys():
            label = kwargs["label"]
        self._plot(ax, type_, means, errors, confidence, label)

        # If another model has been specified, plot it as well
        if self.checkpoint_files2 is not None:
            label2 = ""
            means, errors = self._calculate_values(self.checkpoint_files2,
                                                   self.checkpoint_dir2, type_)
            if "label2" in kwargs.keys():
                label2 = kwargs["label2"]
            self._plot(ax, type_, means, errors, confidence, label2)

        # Set the x/y limits
        if x != (None, None):
            ax.set_xlim(x[0], x[1])
        if y!= (None, None):
            ax.set_ylim(y[0], y[1])

        # Adjust figure text
        ax.set_title(f"Learning Curve ({type_})", fontsize=18)
        ax.set_xlabel("epochs", fontsize=15)
        ax.set_ylabel(type_, fontsize=15)
        ax.legend()

    def _plot(self, ax, type_, means, errors, confidence, label):
        """
        Plots the mean points with a confidence interval about them

        Parameters
        ----------
        ax : plt.Axes
            The axis to plot on
        type_ : str
            The type of learning curve to plot
        means : np.array of float
            The mean values to plot
        errors : np.array of float
            The standard errors of the values to plot
        confidence : float
            The confidence level for the confidence plot
        """
        errors *= confidence

        # spline = make_interp_spline(np.arange(means.shape[0]), means)
        # x_new = np.linspace(1, 40, 300)
        # y_new = spline(x_new)
        # ax.plot(x_new, y_new)

        ax.plot(means, label=label + " " + type_)

        if not confidence < 0.01:
            ax.fill_between(np.arange(means.shape[0]), means - errors,
                            means + errors, alpha=0.2, antialiased=True,
                            label=str(confidence) + " confidence interval")

    def _calculate_values(self, checkpoint_files, checkpoint_dir, type_):
        """
        Calculates the values to plot, which consist of the mean values of
        all learning curves for each trained model, together with the
        standard errors.

        Parameters
        ----------
        checkpoint_files : list of str
            The list of checkpoint files which hold the learning curves for
            each separate model
        checkpoint_dir : str
            The directory which contains the checkpoint files
        type_ : str
            The type of learning curve to plot

        Returns
        -------
        tuple of np.array of float
            A tuple consisting of the mean values to plot, together with the
            standard errors
        """
        points = []
        for file in checkpoint_files:
            checkpoint = load(os.path.join(checkpoint_dir, file))
            points.append(checkpoint["lc"][type_])

        points = np.stack(points, axis=0)
        means = np.mean(points, axis=0)
        errors = np.std(points, axis=0) / np.sqrt(len(self.checkpoint_files))

        return means, errors

