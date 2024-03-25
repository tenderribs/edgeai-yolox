# YOLOX based Models
This repository is a fork of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). This contains the enhancements of the YOLOX repository for supporting additional tasks and embedded friendly ti_lite models.


## Installation

### Step 1: Install Drivers.


My setup is based on RTX 4070 w/ Ada Lovelace arch on Ubuntu 20.04. CPU is AMD R7 3700x.

There are two options for installation.

#### Option 1 Manual Installation

- [Install GPU drivers](https://www.nvidia.com/Download/index.aspx)
- [Install Cuda](https://developer.nvidia.com/cuda-downloads)


Here is what I am running:

```sh
$ nvidia-smi
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce RTX 4070        Off |   00000000:2D:00.0 Off |                  N/A |
    |  0%   35C    P8              7W /  200W |      19MiB /  12282MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Tue_Feb_27_16:19:38_PST_2024
    Cuda compilation tools, release 12.4, V12.4.99
    Build cuda_12.4.r12.4/compiler.33961263_0
```

#### Option 2:

Use the vscode devcontainer contained in the repo and just run the entire project within a container. No GPU driver setup required.

### Step 2: Install YOLOX.

Regardless of the previous driver installation, you have to go through this too.

```sh
./setup.sh
```

### Step 3: Install [pycocotools](https://github.com/cocodataset/cocoapi).

```sh
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### Step 4: Download Datasets
