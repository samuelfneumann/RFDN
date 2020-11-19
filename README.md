# Codebase
The codebase was taken from [Residual Feature Distillation Network for Lightweight Image Super-Resolution](https://github.com/njulj/RFDN).
I am adapting the code from this link.

# Directory Hierarchy and dataFilenames.py
The directory hierarchy must be of the form specified in this section in order
to properly work with the dataFilenames.py file to generate the filenames
dictionaries used for training and evaluation. For all classes and function,
the `data_dir` argument stores the absolute path to the directory
containing the data. This directory should have two files
in it `dataFilenames.bin` and `valFilenames.bin`, each of which storing
a dictionary of (LR, HR) absolute filenames. These dictionaries can
be created using the dataFilenames.py script. The directory
structure for the data directory must be of the following form:

```
data_dir
├── train
│   ├── HR
│   │   └── DIV2K_train_HR
|   |       └── HR TRAINING FILES
│   └── LR
│       └── DIV2K_train_LR_bicubic
│           └── X2
|               └── LR TRAINING FILES
└── val
    ├── HR
    │   └── DIV2K_valid_HR
    |       └── HR VALIDATION FILES
    └── LR
        └── DIV2K_valid_LR_bicubic
            └── X2
                └── LR VALIDATION FILES
```

Given such a directory hierarchy, you can run the script dataFilenames.py, with
a single command line argument, which is the absolute path to the data
directory `data_dir` as above. This script will create the necessary dictionaries
for training and validation and save these dictionaries in `data_dir`. Then,
whenever you use a function or class which requires the `data_dir` parameter,
you can specify the `data_dir` above. The functions and classes will then
look in this directory for the two dictionaries created by dataFilenames.py
and use these dictionaries for training and evaluation.

## Checkpoints
A Trainer object will save a checkpoint after each epoch. This checkpoint can
the be re-loaded into a Trainer at train time in order to start training from
where you last left off. In addition, this checkpoint file can be passed to an
Evaluate object which will then evaluate the checkpoint-ed, trained model
by creating learning curves, or performing predictions on the validation data.

### Contents of Checkpoints
A checkpoint is a Python dictionary with the following key-value pairs:
* **epoch**: the epoch last trained on
* **lr**: the current learning rate for the trained model, utilized if the
        model is trained from the current state
* **model_param**: the model parameters
* **optim_param**: the optimizer parameters, utilized if the model is trained
                    from the current state
* **lc**: the data stored during training used to generate a learning curve

The object stored as the value for the **lc** key is another Python dictionary
with the following key-value pairs:
* **psnr**: the average PSNR value for _X_ randomly sampled validation data
            instances after each epoch of training
* **ssim**: the average SSIM value for _X_ randomly sampled validation data
            instances after each epoch of training
* **loss**: the average loss per mini-batch for the epoch
* **val_items**: the _X_ above

# Trainer
The `Trainer` class takes care of training a network.
Given a model, checkpoint file, data directory, learning rate, and
division factor for the learning rate, a Trainer object can be created by:
```
train = Trainer(model, checkpoint_file, data_dir, learning_rate, division_factor)
```
The `checkpoint_file` parameter is an absolute path to the file at which you
would like to save checkpoints of the model during the training process. The
learning curve data is also saved to this file.
Once this object has been created, you can run the train() method to begin
training the model. The train() method takes in a single mandatory parameter,
the number of epochs to train for. The train() method takes an optional parameter,
`load`. If `True` the checkpoint file will be loaded and training started from
the last checkpoint. If `False`, the checkpoint file will not be loaded. In
either case, checkpoints will be saved to the same directory as the specified
checkpoint file. To train:
```
train.train(epochs)
```

# Evaluate
The Evaluate class is involved in evaluating a trained network. The
`data_dir` instance variable stores the absolute path to the directory containing
the data dictionaries as specified above, in order to be used for
validation metrics. A new model is passed in, and based
on the checkpoint file from the Trainer object, the model parameters
are initialized. The Evaluation class can then find the average PSNR or SSIM on
the validation data, as well as plot the learning curves generated from the
training process. Additionally, the Evaluate object can show predicted images
from the network and comparisons between the network and bicubic interpolation
or comparisons between the network and the high resolution labels.

# Compare
The Compare class will compare two trained networks. To create a Compare object,
you specify two models and their associated checkpoint files, as well as the
data directory as specified above:
```
comp = Compare(model1, model2, checkpoint_file1, checkpoint_file2, data_dir)
```
After you have created this class, you can run a number of functions to
compare the two networks. `plot_lc()` will plot the models' learning curves
on the same axis. `predict()` will upsample an image and produce the PSNR and
SSIM values for each network. `get_values()` will get the average PSNR, SSIM,
and inference time over all validation data instances for each model.
`compare_patches()` will plot predicted patches for an upscaled image
for each model side-by-side.

# Jupyter Notebook Files
Jupyter notebook files were used to train and evaluate models. In addition, they
were used to create figures, compare model outputs, generate data for tables,
generate and analyze learning curves, etc. The Python files held the main
functionality, and the Jupyter notebook files were used to run that functionality.
The following is a list of Jupyter notebook files and how they were used:
1. `AvgLearningCurves.ipynb` was used to generate average learning curves over
multiple trained networks and analyze the average performace.
2. `Compare.ipynb` was used to compare to networks at a time, usually RFDN and RFDN1.
3. `eval.ipynb` was used to evaluate a single network, printing its learning curve
and generating output as well as generating performance metrics.
4. `Figures.ipynb` was used to create all figures for the report.
5. `TablesAndMetrics.ipynb` was used to generate all table data and metrics for
the report.
6. `train.ipynb` was used to train all networks.

# Python files
Here, we outline some of the Python files and what is contained in each file:
1. `BaseN.py` contains the base network used in the ablation study.
2. `block.py` contains all the convolutional blocks for all networks.
3. `compare.py` contains the source code for the Compare object for comparing
two networks.
4. `dataFilenames.py` is a script which will generate the data filename
dictionaries.
5. `evaluate.py` contains the source code for the Evaluate object for evaluating
a single network.
6. `interpolcation.py` contains a class that upscales images via bicubic
interpolation
7. `lc.py` contains the source code for the LearningCurve class, which generates
average learning curves for a set of trained networks.
8. `RFDN.py` contains the RFDN and RFDN1 networks
9. `train.py` contains the source code for the Trainer class for training a
network
10. `utils_image.py` contains utilities for working with images.

# Results
Here, we outline some patches of the resulting predictions.

## RFDN1 vs Bicubic Interpolation
![RFDN1vsInterpolation](../../Report/images/84-RFDN1-Interp.png)

## RFDN vs RFDN1
![RFDNvsRFDN1](../../Report/images/84-RFDN-RFDN1.png)
![RFDNvsRFDN1_2](../../Report/images/84-RFDN-RFDN1_button.png)
![RFDNvsRFDN1_3](../../Report/images/56-RFDN-RFDN1.png)

## RFDN vs High Resolution Label
![RFDN1-HR](../../Report/images/84-RFDN1-HR.png)

# To Do Before Handing in
- [ ] Remove checking if we are on localhost or not to determine LR-HR dictionary filenames
- [ ] Have only dataFilenames.bin and valFilenames.bin in the Data folder, we don't
need any references to Google Colab
- [ ] Remove second copy of util files which were needed for Colab. Only keep util
files in the utils folder.
- [ ] Fix the files to import the util files (above) properly
- [ ] Add documentation to block.py