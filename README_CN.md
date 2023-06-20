# ic

这个仓库包含图片分类用到的一些工具和模型（CNN，ViT 和 [DKL 模型](https://github.com/cornellius-gp/gpytorch/blob/master/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.ipynb)）.

## 用法

1. 要求:
    - Python >= 3.10
2. 按照[官方文档](https://pytorch.org/get-started/locally/)安装 PyTorch.
3. 安装 [d2l](https://github.com/d2l-ai/d2l-zh):
    - 克隆 d2l 仓库:
        ```bash
        git clone https://github.com/d2l-ai/d2l-zh && cd d2l-zh
        ```
    - 移除 `setup.py` 里对依赖项的版本要求.
    - 使用 pip 安装 d2l：
        ```bash
        pip install .
        ```
4. 安装其他依赖项:
    ```bash
    cd /path/to/ic/ && pip install -r requirements.txt
    ```
5. 在 Python 脚本中添加仓库路径:
    ```python
    import sys
    sys.path.append('/path/to/ic/')
    ```
6. 您即可导入此包:
    ```python
    import ic
    ```
