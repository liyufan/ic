# ic

[中文文档](README_CN.md)

This repository contains the tools and models (CNN, ViT and [DKL Model](https://github.com/cornellius-gp/gpytorch/blob/master/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.ipynb)) for image classification.

## Usage

1. Requirements:
    - Python >= 3.10
2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/locally/).
3. Install [d2l](https://github.com/d2l-ai/d2l-en):
    - Clone the d2l repository:
        ```bash
        git clone https://github.com/d2l-ai/d2l-en && cd d2l-en
        ```
    - Delete version specification of requirements in `setup.py`.
    - Install d2l using pip:
        ```bash
        pip install .
        ```
4. Install other requirements:
    ```bash
    cd /path/to/ic/ && pip install -r requirements.txt
    ```
5. Append the path of this repository in python scripts:
    ```python
    import sys
    sys.path.append('/path/to/ic/')
    ```
6. You can then import this package:
    ```python
    import ic
    ```
