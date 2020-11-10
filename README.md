# Codebase
The codebase was taken from [Residual Feature Distillation Network for Lightweight Image Super-Resolution](https://github.com/njulj/RFDN).
I am adapting the code from this link.

# Trainer
The module involved in training a network. The `data_dir` instance variable
stores the absolute path to the directory containing the data. This directory
should have two files in it `dataFilenames.bin` and `valFilenames.bin`, each of
which storing a dictionary of (LR, HR) filenames. The directory structure for
the data directory must be of the following form:

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

The `checkpoint_file` parameter is an absolute path to the file at which you
would like to save checkpoints of the model during the training process. The
learning curve data is also saved to this file.

## Checkpoints
A Trainer object will save a checkpoint after each epoch. This checkpoint can
the be re-loaded into a Trainer at train time in order to start training from
where you last left off. In addition, this checkpoint file can be passed to an
Evaluate object which will then evaluate the checkpoint-ed, trained model
by creating learning curves, or performing predictions on the validation data.

### Contents of Checkpoints
A checkpoint is a Python dictionary with the following key-value pairs:
    **epoch**: the epoch last trained on
    **lr**: the current learning rate for the trained model, utilized if the
            model is trained from the current state
    **model_param**: the model parameters
    **optim_param**: the optimizer parameters, utilized if the model is trained
                     from the current state
    **lc**: the data stored during training used to generate a learning curve

The object stored as the value for the **lc** key is another Python dictionary
with the following key-value pairs:
    **psnr**: the average PSNR value for _X_ randomly sampled validation data
              instances after each epoch of training
    **ssim**: the average SSIM value for _X_ randomly sampled validation data
              instances after each epoch of training
    **loss**: the average loss per mini-batch for the epoch
    **val_items**: the _X_ above

# Evaluate
The evaluate class is involved in evaluating a trained network. The
`data_dir` instance variable stores the absolute path to the directory containing
the data, in order to be used for validation metrics. A new model is passed in,
and based on the checkpoint file from the Trainer object, the model parameters
are initialized. The Evaluation class can then find the average PSNR or SSIM on
the validation data, as well as plot the learning curves generated from the
training process.