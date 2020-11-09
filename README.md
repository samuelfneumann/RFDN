# RFDN
Code for our paper [Residual Feature Distillation Network for Lightweight Image Super-Resolution](https://arxiv.org/abs/2009.11551)

We won the first place in [AIM 2020 efficient super-resolution challenge](https://data.vision.ee.ethz.ch/cvl/aim20/), the accepted workshop paper and code will be released soon.

The model files are uploaded! You can use the [EDSR framework](https://github.com/thstkdgus35/EDSR-PyTorch) to train our RFDN and use the [AIM test code](https://github.com/znsc/MSRResNet) to reproduce results in the [AIM challenge paper](https://arxiv.org/abs/2009.06943)

**The pretrained models and test codes are uploaded, now you can run test.py to get results in the challenge.**


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
would like to save checkpoints of the model during the training process.

If applicable, the Trainer object will save the learning curve data in the
current working directory.