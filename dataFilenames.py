# Import modules
import os
import sys
import pickle


# Function definitions
def create_train_dictionary(data_dir):
    """
    Creates the training dictionary used for training the model. The dictionary
    keys are the LR image absolute filenames, and the values are the HR image
    absolute filenames.

    Parameters
    ----------
    data_dir : str
        The absolute path to the data_dir file as specified in the directory
        hierarchy structure of the README.md file.

    Raises
    ------
    ValueError
        If the data directory does not exist
    """
    if not os.path.isdir(data_dir):
        raise ValueError("error: no such directory exists")

    # Get LR train filenames
    train_lr_dir = os.path.join(data_dir, "train/LR/DIV2K_train_LR_bicubic/X2")
    train_lr_filenames = sorted(os.listdir(train_lr_dir))
    train_lr_filenames = list(map(lambda x: os.path.join(train_lr_dir, x),
                                  train_lr_filenames))

    # Get HR train filenames
    train_hr_dir = os.path.join(data_dir, "train/HR/DIV2K_train_HR")
    train_hr_filenames = sorted(os.listdir(train_hr_dir))
    train_hr_filenames = list(map(lambda x: os.path.join(train_hr_dir, x),
                                  train_hr_filenames))

    # Create the data dictionary and save
    train_dict = dict(zip(train_lr_filenames, train_hr_filenames))
    with open(os.path.join(data_dir, "dataFilenames.bin"), "wb") as train_file:
        pickle.dump(train_dict, train_file)


def create_val_dictionary(data_dir):
    """
    Creates the validation dictionary used for evaluation of the model.
    The dictionary keys are the LR image absolute filenames, and the
    values are the HR image absolute filenames.

    Parameters
    ----------
    data_dir : str
        The absolute path to the data_dir file as specified in the directory
        hierarchy structure of the README.md file.

    Raises
    ------
    ValueError
        If the data directory does not exist
    """
    if not os.path.isdir(data_dir):
        raise ValueError("error: no such directory exists")

    # Get LR val filenames
    val_lr_dir = os.path.join(data_dir, "val/LR/DIV2K_valid_LR_bicubic/X2")
    val_lr_filenames = sorted(os.listdir(val_lr_dir))
    val_lr_filenames = list(map(lambda x: os.path.join(val_lr_dir, x),
                                val_lr_filenames))

    # Get HR val filenames
    val_hr_dir = os.path.join(data_dir, "val/HR/DIV2K_valid_HR")
    val_hr_filenames = sorted(os.listdir(val_hr_dir))
    val_hr_filenames = list(map(lambda x: os.path.join(val_hr_dir, x),
                                val_hr_filenames))

    # Create the data dictionary and save
    val_dict = dict(zip(val_lr_filenames, val_hr_filenames))
    with open(os.path.join(data_dir, "valFilenames.bin"), "wb") as val_file:
        pickle.dump(val_dict, val_file)


# Script
if __name__ == "__main__":
    data_dir = sys.argv[1]
    create_train_dictionary(data_dir)
    create_val_dictionary(data_dir)
