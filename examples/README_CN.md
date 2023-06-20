# 示例

此处包含图片分类的示例代码。

## 训练配置
此仓库使用一个统一的配置，您应该在训练脚本中指定它：
```python
config = {
    # 训练 epoch 数
    "n_epochs": 200,
    # 传递给 DataLoader 的参数
    "data_loader": {"batch_size": 256, "num_workers": 16, "pin_memory": True},
    # 优化器名称
    "optimizer": "Adam",
    # 传递给优化器的参数
    "optim_hparas": {"lr": 1e-3, "weight_decay": 1e-4},
    # 损失函数
    "criterion": nn.CrossEntropyLoss(),
    # 是否使用预训练模型，
    # 对于 CNN 此处为布尔值，对于 HuggingFace ViT 为字符串，
    # 例如 "google/vit-base-patch16-224-in21k"
    "weights": False,
}
```

## MoCo（无监督方法）
### MoCo 预训练
为了对 [MoCo](https://arxiv.org/abs/1911.05722) 预训练，下载其[源码](https://github.com/facebookresearch/moco)，在 [moco.sh](moco.sh) 中指定模型名称并运行：
```bash
bash moco.sh
```
如果您想使用 [DenseNet](https://arxiv.org/abs/1608.06993) 作为特征提取器，您需要修改 MoCo 的源码以支持它，即将所有 `"fc"` 改为 `"classifier"`.如果您想使用 [ViT](https://arxiv.org/abs/2010.11929)，请参考 [MoCo v3](https://github.com/facebookresearch/moco-v3).
### 线性分类
至于线性分类，在 [val.sh](val.sh) 中指定模型名称和保存模型的位置并运行：
```bash
bash val.sh
```
您可以通过修改 `main_lincls.py` 中的 `main_worker` 函数来在 [val.sh](val.sh) 中更改模型的类数：
```python
model = models.__dict__[args.arch](num_classes=10)
```
### SVM 和 GP 作为分类器
为了在冻结的特征提取器（使用MoCo预训练得到）上使用 SVM 或 GP 分类，参照 [moco.ipynb](moco.ipynb).

## 监督方法
参照 [dnn.ipynb](dnn.ipynb) 来训练一个常用的监督模型；参照 [gp.ipynb](gp.ipynb) 来训练一个 DKL 模型。
