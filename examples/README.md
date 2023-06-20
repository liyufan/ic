# Examples

[中文文档](README_CN.md)

This contains code for image classification.

## Training config
This repository uses a unified config, you should include it in your training script:
```python
config = {
    # Nums of epochs to train
    "n_epochs": 200,
    # Kwargs passed to DataLoader
    "data_loader": {"batch_size": 256, "num_workers": 16, "pin_memory": True},
    # Name of the optimizer
    "optimizer": "Adam",
    # Kwargs passed to the optimizer
    "optim_hparas": {"lr": 1e-3, "weight_decay": 1e-4},
    # Loss function
    "criterion": nn.CrossEntropyLoss(),
    # Whether to use pretrained model,
    # boolean for CNN, string for HuggingFace ViT,
    # e.g., "google/vit-base-patch16-224-in21k"
    "weights": False,
}
```

## MoCo (unsupervised)
### MoCo pretraining
To pretrain [MoCo](https://arxiv.org/abs/1911.05722), download its [source](https://github.com/facebookresearch/moco), specify the model name in [moco.sh](moco.sh) then run:
```bash
bash moco.sh
```
If you want to use [DenseNet](https://arxiv.org/abs/1608.06993) as the backbone, you need to modify the source code of MoCo to support it, i.e., change all `"fc"` to `"classifier"`. If you want to use [ViT](https://arxiv.org/abs/2010.11929), refer to [MoCo v3](https://github.com/facebookresearch/moco-v3).
### Linear classification
For linear classification, specify the model name and checkpoint location in [val.sh](val.sh) then run:
```bash
bash val.sh
```
You can change number of classes of model in [val.sh](val.sh) by modifying the function `main_worker` in `main_lincls.py`:
```python
model = models.__dict__[args.arch](num_classes=10)
```
### SVM or GP as classifier
For classification using SVM or GP on a frozen feature extractor (pretrained using MoCo), refer [moco.ipynb](moco.ipynb)

## Supervised
Refer [dnn.ipynb](dnn.ipynb) for training a normal supervised model and [gp.ipynb](gp.ipynb) for training a DKL model.
