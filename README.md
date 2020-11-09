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