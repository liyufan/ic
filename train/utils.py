# Copyright (c) 2022-2023 Johnson Lee. All rights reserved.

# A collection of tools used for image classification.
# Functions with 'dkl' in their names are for DKL models.


import builtins
import operator
import random
import sys
import warnings
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from d2l import torch as d2l
from gpytorch import settings
from gpytorch.likelihoods import Likelihood
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor, nn
from torch._dynamo import OptimizedModule
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.types import Device
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Not use `from tqdm.auto import tqdm` because it yields error like this
# when using with `torch.multiprocessing`:
"""
Exception in thread QueueFeederThread:
Traceback (most recent call last):
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/queues.py", line 239, in _feed
Exception in thread QueueFeederThread:
Traceback (most recent call last):
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/queues.py", line 239, in _feed
Exception ignored in: <function _ConnectionBase.__del__ at 0x7f92f8b9cc20>
Traceback (most recent call last):
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/connection.py", line 132, in __del__
    reader_close()
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/connection.py", line 177, in close
    self._close()
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/connection.py", line 360, in _close
    reader_close()
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/connection.py", line 177, in close
    self._close()
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/connection.py", line 360, in _close
    self._close()
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/connection.py", line 360, in _close
    _close(self._handle)
OSError: [Errno 9] Bad file descriptor
    _close(self._handle)
OSError: [Errno 9] Bad file descriptor

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "~/anaconda3/envs/torch/lib/python3.11/threading.py", line 1038, in _bootstrap_inner
    self.run()
  File "~/anaconda3/envs/torch/lib/python3.11/threading.py", line 975, in run
    _close(self._handle)
OSError: [Errno 9] Bad file descriptor

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "~/anaconda3/envs/torch/lib/python3.11/threading.py", line 1038, in _bootstrap_inner
    self._target(*self._args, **self._kwargs)
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/queues.py", line 271, in _feed
    self.run()
  File "~/anaconda3/envs/torch/lib/python3.11/threading.py", line 975, in run
    queue_sem.release()
ValueError: semaphore or lock released too many times
    self._target(*self._args, **self._kwargs)
  File "~/anaconda3/envs/torch/lib/python3.11/multiprocessing/queues.py", line 271, in _feed
    queue_sem.release()
ValueError: semaphore or lock released too many times
"""
from tqdm.autonotebook import tqdm

from ..models import DKLModel


def setattr_nested(__obj: object, __path: str, __value: Any, /):
    r"""Accept a dotted path to a nested attribute to set."""
    assert (
        "." in __path
    ), "Path must be a dotted path to a nested attribute. Use `setattr` for non-nested attributes."
    path, _, target = __path.rpartition(".")
    obj = operator.attrgetter(path)(__obj)
    setattr(obj, target, __value)


def ensure_reproducibility(seed: int = 0) -> int:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    warnings.warn(
        'You have chosen to seed training. '
        'This will turn on the CUDNN deterministic setting, '
        'which can slow down your training considerably! '
        'You may see unexpected behavior when restarting '
        'from checkpoints.'
    )
    return seed


def create_data_loaders(
    config: dict, train_set: Dataset, test_set: Dataset, distributed: bool = False
) -> tuple[DataLoader, DataLoader]:
    r"""Return a tuple of train and test data loaders.

    Args:
        config: A dictionary containing configuration for data loaders.
        train_set: A dataset for training.
        test_set: A dataset for testing.
        distributed: Whether to use distributed training.
    """
    config_data_loader = config["data_loader"]
    if "shuffle" in config_data_loader:
        warnings.warn(
            "`shuffle` is set in config['data_loader']. "
            "This will be overridden by `shuffle=True` for `train_loader` "
            "and `shuffle=False` for `test_loader`."
        ) if not distributed else warnings.warn(
            "`shuffle` is set in config['data_loader']. "
            "This will be overridden by `shuffle=DistributedSampler(train_set)` "
            "for `train_loader` and `shuffle=False` for `test_loader`."
        )
        del config_data_loader["shuffle"]
    train_sampler, test_sampler = (
        (DistributedSampler(train_set), DistributedSampler(test_set, shuffle=False))
        if distributed
        else (None, None)
    )
    train_loader = DataLoader(
        train_set,
        **config_data_loader,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_set, **config_data_loader, shuffle=False, sampler=test_sampler
    )
    return train_loader, test_loader


def _forward(X: Tensor, model: nn.Module) -> Tensor:
    r"""Return model output.

    Call this after `model.to(device)` or `model.cuda()`.
    """
    if next(model.parameters()).device != X.device:
        raise RuntimeError(
            "Model and data must be on the same device. "
            "Call this after `model.to(device)` or `model.cuda()`."
        )
    t = model(X)
    return t.logits if hasattr(t, "logits") else t  # For huggingface transformers


def model_fn(
    X: Tensor,
    y: Tensor,
    model: nn.Module,
    criterion: nn.Module | None = None,
) -> tuple[Tensor | None, Tensor]:
    r"""Return loss and accuracy for a batch.

    Call this after `model.to(device)` or `model.cuda()`.

    Note, accuracy returned may be on cuda.
    """
    outs = _forward(X, model).squeeze()

    if criterion:
        loss = criterion(outs, y)
    else:
        loss = None

    preds = outs.argmax(-1)
    accuracy = torch.mean((preds == y).float())

    return loss, accuracy


def topk_accuracy(
    batch: list[Tensor],
    model: nn.Module,
    device: Device,
    k: int = 5,
) -> Tensor:
    r"""Return top-k accuracy for a batch.
    Note, accuracy returned may be on cuda."""
    X, y = batch
    X, y = X.to(device), y.to(device)

    outs = _forward(X, model)

    preds = outs.topk(k, dim=-1)[1]
    return torch.mean((preds == y.unsqueeze(-1)).any(-1).float())


def topk_accuracies(
    batch: list[Tensor],
    model: nn.Module,
    device: Device,
    ks: list[int] = [3, 5],
) -> Tensor:
    r"""Return a tensor of top-k accuracies for each k in ks.
    Note, accuracies returned may be on cuda."""
    accuracies = torch.tensor([], device=device)

    for k in ks:
        accuracies = torch.cat(
            (accuracies, topk_accuracy(batch, model, device, k).unsqueeze(0))
        )
    return accuracies


def get_model_last_layer_name(model: nn.Module) -> str:
    r"""Return the name of the last layer of a model."""
    # Returned value may be nested if model contains another one, e.g. `resnet.fc` rather than `fc`.
    # Consider using `operator.attrgetter` rather than `getattr` to access the last layer
    return list(model.named_modules())[-1][0]


def get_model_last_layer(model: nn.Module) -> nn.Module:
    r"""Return the last layer of a model."""
    # or `operator.attrgetter(get_model_last_layer_name(model))(model)`
    return list(model.modules())[-1]


def get_feature_dim(model: nn.Module) -> int:
    r"""Return the dimension of the feature extracted by a model."""
    linear = get_model_last_layer(model)
    assert type(linear) == nn.Linear, (
        "Last layer must be a linear layer."
        "Your model contains a feature extractor only, i.e., fully connected layer is not present."
        "For ViTModel, use `vit.config.hidden_size` instead."
    )
    return linear.in_features


def set_num_classes(model: nn.Module, num_classes: int):
    r"""Set the number of classes for a model."""
    last_layer_name = get_model_last_layer_name(model)
    feature_dim = get_feature_dim(model)
    setattr_nested(
        model,
        last_layer_name,
        nn.Linear(feature_dim, num_classes),
    ) if "." in last_layer_name else setattr(
        model,
        last_layer_name,
        nn.Linear(feature_dim, num_classes),
    )


def create_optimizer(config: dict, model: nn.Module) -> Optimizer:
    r"""Return an optimizer for a model with a given configuration."""
    # For fine-tuning, we usually set the learning rate of the last layer to be 10 times larger
    if "weights" in config:
        last_layer_name = get_model_last_layer_name(model)
        params_1x = [
            param
            for name, param in model.named_parameters()
            if name
            not in [
                f"{last_layer_name}.weight",
                f"{last_layer_name}.bias",
            ]
        ]
        lr = config["optim_hparas"]["lr"]
        wd = (
            config["optim_hparas"]["weight_decay"]
            if "weight_decay" in config["optim_hparas"]
            else 0
        )
        return getattr(torch.optim, config["optimizer"])(
            [
                {"params": params_1x, "lr": lr},
                {
                    "params": get_model_last_layer(model).parameters(),
                    "lr": lr * 10,
                },
            ],
            lr=lr,
            weight_decay=wd,
        )
    # For normal training
    else:
        return getattr(torch.optim, config["optimizer"])(
            model.parameters(), **config["optim_hparas"]
        )


def test(model: nn.Module, test_loader: DataLoader, device: Device) -> float:
    r"""Return accuracy for a test set.
    Note the following code is wrong when
    set `drop_last=False` in `DataLoader`, because the last batch may not be full,
    using `len(test_loader)` is not correct.
        >>> model.eval()
        >>> running_accuracy = 0.0
        >>> with torch.no_grad():
        >>>     for batch in test_loader:
        >>>         X, y = batch[0].to(device), batch[1].to(device)
        >>>         _, acc = model_fn(X, y, model)
        >>>         running_accuracy += acc.item()
        >>> return running_accuracy / len(test_loader)
    """
    y_true, y_pred = model_y_true_y_pred(model, test_loader, device, False)
    return torch.mean((y_true == y_pred).float()).item()


def convert_tensor_to_list(x: Tensor) -> list[float]:
    m: NDArray = x.detach().cpu().numpy()
    return m.tolist()


def test_topk(
    model: nn.Module, test_loader: DataLoader, device: Device, ks: list[int] = [3, 5]
):
    raise RuntimeError("This function is not correct.")
    # TODO: Fix this, see `test` function for reference.
    # The problem is that the last batch may not be full.
    r"""Return a list of top-k accuracies for each k in ks."""
    model.eval()
    running_accuracies = torch.tensor([0.0] * len(ks))
    with torch.no_grad():
        for batch in test_loader:
            accs = topk_accuracies(batch, model, device, ks)
            running_accuracies += accs.detach().cpu()
    return convert_tensor_to_list(running_accuracies / len(test_loader))


def create_animator(n_epochs: int) -> d2l.Animator:
    r"""Return an animator for training and testing losses and accuracies."""
    return d2l.Animator(
        xlabel="epoch",
        xlim=[1, n_epochs],
        ylim=[0, 1],
        legend=["train loss", "train acc", "test acc"],
    )


def train_all_gpu(
    config: dict, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader
):
    r"""Train a model with DataParallel on all GPUs.

    Consider use `train_distributed` instead, see
    https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
    """
    assert not isinstance(
        model, DKLModel
    ), "DKLModel is not supported. Use `gpytorch.kernels.MultiDeviceKernel` instead."
    assert not next(
        model.parameters()
    ).is_cuda, "Model must be on CPU, or it will cause glitches when wrapped by `nn.DataParallel`."
    n_batches = len(train_loader)
    n_epochs = config["n_epochs"]
    animator = create_animator(n_epochs)
    n_gpus = torch.cuda.device_count()
    gpus = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
    device = gpus[0]
    dp = nn.DataParallel(model, device_ids=gpus).to(device)
    optimizer = create_optimizer(config, dp)
    criterion = config["criterion"]
    for epoch in range(n_epochs):
        metric = d2l.Accumulator(3)
        dp.train()
        for i, batch in enumerate(train_loader):
            X, y = batch[0].to(device), batch[1].to(device)
            l, acc = model_fn(X, y, dp, criterion)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            metric.add(l.item(), acc.item(), 1)
            if (i + 1) % 2 == 0 or i == n_batches - 1:
                animator.add(
                    epoch + (i + 1) / n_batches,
                    (metric[0] / metric[2], metric[1] / metric[2], None),
                )
        test_acc = test(dp, test_loader, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(
        f"loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[2]:.3f}, test acc {test_acc:.3f}"
    )


def train(
    config: dict,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: Device,
    distributed: bool = False,
):
    r"""Train a model on a device with a given configuration.

    Remember to move the model to the device before calling this function.
    """
    assert not isinstance(model, DKLModel), "Use `train_dkl` for DKL models"
    if isinstance(model, DDP):
        assert not isinstance(model.module, DKLModel), "Use `train_dkl` for DKL models"
    n_batches = len(train_loader)
    n_epochs = config["n_epochs"]
    animator = create_animator(n_epochs)
    optimizer = create_optimizer(config, model)
    criterion = config["criterion"].to(device)
    for epoch in range(n_epochs):
        if distributed:
            assert train_loader.sampler is not None, (
                "Distributed training requires a sampler. "
                "Set `distributed=False` for non-distributed training."
            )
            train_loader.sampler.set_epoch(epoch)
        metric = d2l.Accumulator(3)
        model.train()
        for i, batch in enumerate(train_loader):
            X, y = batch[0].to(device), batch[1].to(device)
            l, acc = model_fn(X, y, model, criterion)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            metric.add(l.item(), acc.item(), 1)
            if (i + 1) % 2 == 0 or i == n_batches - 1:
                animator.add(
                    epoch + (i + 1) / n_batches,
                    (metric[0] / metric[2], metric[1] / metric[2], None),
                )
        test_acc = test(model, test_loader, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(
        f"loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[2]:.3f}, test acc {test_acc:.3f}"
    )


def main_worker(
    gpu: int,
    n_gpus: int,
    dist_url: str,
    world_size: int,
    config: dict,
    model: nn.Module,
    train_set: Dataset,
    test_set: Dataset,
    save_folder: str,
):
    r"""Main worker for distributed training."""
    # suppress printing if not master
    if gpu != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    print(f"Use GPU: {gpu} for training")
    rank = 0 * n_gpus + gpu  # rank 0 is the master node
    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=rank
    )
    # https://github.com/pytorch/examples/blob/main/imagenet/main.py
    config["data_loader"]["batch_size"] = int(
        config["data_loader"]["batch_size"] / n_gpus
    )
    config["data_loader"]["num_workers"] = int(
        (config["data_loader"]["num_workers"] + n_gpus - 1) / n_gpus
    )
    train_loader, test_loader = create_data_loaders(
        config, train_set, test_set, distributed=True
    )
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    ddp = DDP(model, device_ids=[gpu])

    compile_model = False
    # `torch.__version__` is an instance of `torch.torch_version.TorchVersion`
    # convert it to a string
    if (
        str(torch.__version__) >= '2'
        and sys.version_info < (3, 11)
        or str(torch.__version__) >= '2.1'
        and sys.version_info >= (3, 11)
    ):
        ddp: OptimizedModule = torch.compile(ddp)
        compile_model = True

    device = torch.device(f"cuda:{gpu}")
    train(config, ddp, train_loader, test_loader, device, distributed=True)

    if rank % n_gpus == 0:
        # We can access attributes of a compiled module directly; For example:
        """
        >> model = torch.nn.Linear(2, 2)
        >> model = torch.compile(model)
        >> model.weight, model.in_features
        so access `OptimizedModule.module` is equivalent to access `DDP.module`
        """
        file = f"{save_folder}/{ddp.module.__class__.__name__}.pth.tar"
        torch.save(ddp.state_dict(), file)
        prefix = "_orig_mod.module." if compile_model else "module."
        print(
            f"Model saved to {file} . "
            "Call `torch.nn.modules.utils."
            f"consume_prefix_in_state_dict_if_present(ddp_state_dict, '{prefix}')` "
            "to remove the prefix before loading the state dict. Example:\n"
            ">>> ckpt = torch.load(f'{ckpt_name}.pth.tar', map_location='cpu')\n"
            ">>> consume_prefix_in_state_dict_if_present(ckpt, 'module.')\n"
            ">>> model.load_state_dict(ckpt)"
        )


# Mod based on https://stackoverflow.com/questions/15411967
def is_notebook() -> bool:
    r"""Return whether the current environment is a Jupyter notebook."""
    try:
        from IPython import get_ipython

        instance = get_ipython()
        if instance is None:  # Standard Python interpreter
            return False
        name = instance.__class__.__name__
        return name == "ZMQInteractiveShell"  # Jupyter notebook or qtconsole
    except ImportError:
        return False  # If IPython isn't installed


def train_distributed(
    dist_url: str,
    config: dict,
    model: nn.Module,
    train_set: Dataset,
    test_set: Dataset,
    save_folder: str = ".",
):
    r"""Train a model with DistributedDataParallel on all GPUs (on current node)."""
    assert not isinstance(model, DKLModel), "DKLModel is not supported."
    assert not isinstance(train_set, DataLoader) and not isinstance(
        test_set, DataLoader
    ), "Pass `Dataset` objects instead of `DataLoader` objects."
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus * 1  # 1 node
    args = (
        n_gpus,
        dist_url,
        world_size,
        config,
        model,
        train_set,
        test_set,
        save_folder,
    )
    mp.start_processes(
        main_worker, args=args, nprocs=n_gpus, start_method='fork'
    ) if is_notebook() else mp.spawn(main_worker, args=args, nprocs=n_gpus)


def model_y_true_y_pred(
    model: nn.Module,
    test_loader: DataLoader,
    device: Device,
    progress: bool = True,
    return_numpy: bool = False,
) -> tuple[Tensor, Tensor] | tuple[NDArray, NDArray]:
    r"""Return a tuple of true labels and predicted labels."""
    assert not isinstance(
        model, DKLModel
    ), "Use `model_y_true_y_pred_dkl` for DKL models"
    model.eval()
    y_true = torch.tensor([])
    y_pred = torch.tensor([])
    with torch.no_grad():
        if progress:
            pbar = tqdm(test_loader)
            for batch in pbar:
                X, y = batch[0].to(device), batch[1]
                y_true = torch.cat((y_true, y))
                y_pred = torch.cat((y_pred, model(X).argmax(dim=1).detach().cpu()))
            pbar.close()
        else:
            for batch in test_loader:
                X, y = batch[0].to(device), batch[1]
                y_true = torch.cat((y_true, y))
                y_pred = torch.cat((y_pred, model(X).argmax(dim=1).detach().cpu()))
    return (y_true, y_pred) if not return_numpy else (y_true.numpy(), y_pred.numpy())


def model_y_true_y_pred_dkl(
    model: DKLModel,
    likelihood: Likelihood,
    test_loader: DataLoader,
    device: Device,
    return_numpy: bool = False,
) -> tuple[Tensor, Tensor] | tuple[NDArray, NDArray]:
    r"""Return a tuple of true labels and predicted labels for a DKL model."""
    model.eval()
    likelihood.eval()
    y_true = torch.tensor([])
    y_pred = torch.tensor([])
    with torch.no_grad(), settings.num_likelihood_samples(16):
        for batch in test_loader:
            X, y = batch[0].to(device), batch[1]
            y_true = torch.cat((y_true, y))
            y_pred = torch.cat(
                (y_pred, likelihood(model(X)).probs.mean(0).argmax(-1).detach().cpu())
            )
    return (y_true, y_pred) if not return_numpy else (y_true.numpy(), y_pred.numpy())


def create_optimizer_dkl(
    config: dict, model: DKLModel, likelihood: Likelihood | DDP
) -> Optimizer:
    r"""Return an optimizer for a DKL model with a given configuration."""
    optim_hparas = config["optim_hparas"]
    lr = optim_hparas["lr"]
    return getattr(torch.optim, config["optimizer"])(
        [
            {'params': model.feature_extractor.parameters(), **optim_hparas},
            {'params': model.gp_layer.hyperparameters(), 'lr': 1e-4},
            {'params': model.gp_layer.variational_parameters(), 'lr': 1e-2},
            {'params': likelihood.parameters(), 'lr': 1e-2},
        ],
        lr=lr,
        weight_decay=0,
    )


def model_fn_dkl(
    X: Tensor,
    y: Tensor,
    model: DKLModel | DDP,
    likelihood: Likelihood | DDP,
    mll: nn.Module | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Return loss and accuracy for a batch for a DKL model."""
    output = model(X)
    loss = -mll(output, y)
    output = likelihood(output)
    acc = (output.probs.mean(0).argmax(-1) == y).float().mean()
    return loss, acc


def test_dkl(
    model: DKLModel | DDP,
    likelihood: Likelihood | DDP,
    test_loader: DataLoader,
    device: Device,
) -> float:
    r"""Return accuracy for a test set for a DKL model."""
    model.eval()
    likelihood.eval()
    correct = 0
    with torch.no_grad(), settings.num_likelihood_samples(16):
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = likelihood(model(X))
            pred = output.probs.mean(0).argmax(-1)
            correct += pred.eq(y.reshape(pred.size())).cpu().sum()
    return correct / len(test_loader.dataset)


def train_dkl(
    config: dict,
    model: DKLModel | DDP,
    likelihood: Likelihood | DDP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: Device,
    distributed: bool = False,
):
    r"""Train a DKL model on a device with a given configuration.

    Remember to move the model and likelihood to the device before calling this function.
    """
    if isinstance(model, DDP):
        assert isinstance(
            model.module, DKLModel
        ), "Model wrapped by DDP must be a DKLModel."
    if isinstance(likelihood, DDP):
        assert isinstance(
            likelihood.module, Likelihood
        ), "Likelihood wrapped by DDP must be a Likelihood."
    n_batches = len(train_loader)
    n_epochs = config["n_epochs"]
    animator = create_animator(n_epochs)
    optimizer = (
        create_optimizer_dkl(config, model, likelihood)
        if isinstance(model, DKLModel)
        else create_optimizer_dkl(config, model.module, likelihood)
    )
    mll = config["criterion"]
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs])
    for epoch in range(n_epochs):
        if distributed:
            assert (
                train_loader.sampler is not None
            ), "Distributed training requires a sampler. Set `distributed=False` for non-distributed training."
            train_loader.sampler.set_epoch(epoch)
        metric = d2l.Accumulator(3)
        with settings.use_toeplitz(False):
            model.train()
            likelihood.train()
            with settings.num_likelihood_samples(8):
                for i, batch in enumerate(train_loader):
                    X, y = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    loss, acc = model_fn_dkl(X, y, model, likelihood, mll)
                    loss.backward()
                    optimizer.step()
                    metric.add(loss.item(), acc.item(), 1)
                    if (i + 1) % 2 == 0 or i == n_batches - 1:
                        animator.add(
                            epoch + (i + 1) / n_batches,
                            (metric[0] / metric[2], metric[1] / metric[2], None),
                        )
            test_acc = test_dkl(model, likelihood, test_loader, device)
            animator.add(epoch + 1, (None, None, test_acc))
        scheduler.step()
    print(
        f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[2]:.3f}, test acc {test_acc:.3f}'
    )


def main_worker_dkl(
    gpu: int,
    n_gpus: int,
    dist_url: str,
    world_size: int,
    config: dict,
    model: DKLModel,
    likelihood: Likelihood,
    train_set: Dataset,
    test_set: Dataset,
    save_folder: str,
):
    # suppress printing if not master
    if gpu != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    print(f"Use GPU: {gpu} for training")
    rank = 0 * n_gpus + gpu  # rank 0 is the master node
    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=rank
    )
    config["data_loader"]["batch_size"] = int(
        config["data_loader"]["batch_size"] / n_gpus
    )
    config["data_loader"]["num_workers"] = int(
        (config["data_loader"]["num_workers"] + n_gpus - 1) / n_gpus
    )
    train_loader, test_loader = create_data_loaders(
        config, train_set, test_set, distributed=True
    )
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    likelihood.cuda(gpu)
    ddp_model = DDP(model, device_ids=[gpu])
    ddp_likelihood = DDP(likelihood, device_ids=[gpu])
    device = torch.device(f"cuda:{gpu}")
    train_dkl(
        config,
        ddp_model,
        ddp_likelihood,
        train_loader,
        test_loader,
        device,
        distributed=True,
    )
    print("Finished Training")


def train_distributed_dkl(
    dist_url: str,
    config: dict,
    model: DKLModel,
    likelihood: Likelihood,
    train_set: Dataset,
    test_set: Dataset,
    save_folder: str,
):
    assert not isinstance(train_set, DataLoader) and not isinstance(
        test_set, DataLoader
    ), "Pass `Dataset` objects instead of `DataLoader` objects."
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus * 1  # 1 node
    args = (
        n_gpus,
        dist_url,
        world_size,
        config,
        model,
        likelihood,
        train_set,
        test_set,
        save_folder,
    )
    mp.start_processes(
        main_worker_dkl, args=args, nprocs=n_gpus, start_method='fork'
    ) if is_notebook() else mp.spawn(main_worker_dkl, args=args, nprocs=n_gpus)


def display_confusion_matrix(
    model: nn.Module,
    test_loader: DataLoader,
    device: Device,
    normalize: Literal['true', 'pred', 'all'] | None = None,
    display_labels: ArrayLike | None = None,
):
    r"""Display a confusion matrix for a model."""
    assert not isinstance(model, DKLModel), "Use `display_confusion_matrix_dkl` instead"
    assert normalize in ['true', 'pred', 'all', None], "Invalid value for `normalize`"
    y_true, y_pred = model_y_true_y_pred(model, test_loader, device, return_numpy=True)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize=normalize, display_labels=display_labels
    )
    plt.show()


def display_confusion_matrix_dkl(
    model: DKLModel,
    likelihood: Likelihood,
    test_loader: DataLoader,
    device: Device,
    normalize: Literal['true', 'pred', 'all'] | None = None,
    display_labels: ArrayLike | None = None,
):
    r"""Display a confusion matrix for a DKL model."""
    assert normalize in ['true', 'pred', 'all', None], "Invalid value for `normalize`"
    y_true, y_pred = model_y_true_y_pred_dkl(
        model, likelihood, test_loader, device, return_numpy=True
    )
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize=normalize, display_labels=display_labels
    )
    plt.show()
