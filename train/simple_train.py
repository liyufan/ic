# A simple training tool which output accuracy in stdout (without charts).


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.types import Device


def test(model: nn.Module, test_loader: DataLoader, device: Device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).squeeze()
            numbers, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Testing Accuracy : %.3f %%" % (100 * correct / total))


def train(
    config: dict,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: Device,
):
    model = model.to(device)
    criterion = config["criterion"]
    optimzier = getattr(torch.optim, config["optimizer"])(
        model.parameters(), **config["optim_hparas"]
    )
    epochs = config["n_epochs"]
    for _epoch in range(epochs):
        training_loss = 0.0
        model.train()
        for _step, input_data in enumerate(train_loader):
            image, label = input_data[0].to(device), input_data[1].to(device)
            predict_label = model(image).squeeze()

            loss = criterion(predict_label, label)

            optimzier.zero_grad()
            loss.backward()
            optimzier.step()

            training_loss = training_loss + loss.item()
            if (_step + 1) % 10 == 0:
                print(
                    "[iteration - %3d] training loss: %.3f"
                    % (_epoch * len(train_loader) + _step + 1, training_loss / 10)
                )
                training_loss = 0.0
        print("Epoch %d finished." % (_epoch + 1), end=" ")
        test(model, test_loader, device)
