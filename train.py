import torch
from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy
import copy


def train(model, epochs, loss_func, optimizer, train_dataloader, val_dataloader, device, num_classes=5):
    f1 = MulticlassF1Score(num_classes=num_classes)
    acc = MulticlassAccuracy(num_classes=num_classes)

    train_losses = []
    best_f1 = 0.

    for i in range(epochs):

        batch_loss, n_batches = 0., 0
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device).float(), y.to(device)
            y_hat = model(X)
            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            n_batches += 1

        batch_loss = batch_loss / n_batches
        train_losses.append(batch_loss)

        f1.reset()
        acc.reset()
        model.eval()
        for X, y in val_dataloader:
            X, y = X.to(device).float(), y.to(device)
            with torch.no_grad():
                y_hat = model(X)

            f1.update(y_hat, y)
            acc.update(y_hat, y)

        print(f"Epoch {i}:\n\ttrain loss: {batch_loss}\n\tval f1: {f1.compute().item()}\n\tval acc: {acc.compute().item()}")
        if best_f1 < f1.compute().item():
            best_f1 = f1.compute().item()
            best_model = copy.deepcopy(model)

    return best_model, train_losses