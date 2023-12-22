import torch
from torcheval.metrics import MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassAccuracy, MulticlassConfusionMatrix

import numpy as np
import copy


def evaluate(best_model, test_dataloader, device, num_classes=4):
    metrics = {"f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
               "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
               "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
               "accuracy": MulticlassAccuracy(num_classes=num_classes),
               "conf_mat": MulticlassConfusionMatrix(num_classes=num_classes)
               }

    best_model.eval()
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_hat = best_model(X.float())
    
        for metric in metrics:
            metrics[metric].update(y_hat, y)
        
    for metric in metrics:
        if metric == "conf_mat":
            metrics[metric] = metrics[metric].compute().long()
        else:
            metrics[metric] = metrics[metric].compute()

    return metrics


def train(model, epochs, loss_func, optimizer, train_dataloader, val_dataloader, device, num_classes=5, verbose=False):
    f1 = MulticlassF1Score(num_classes=num_classes)
    f1_t = MulticlassF1Score(num_classes=num_classes)
    acc_t = MulticlassAccuracy(num_classes=num_classes)

    # Initialize variables for tracking best loss and storing losses at each epoch
    best_f1, train_losses, f1s_v, f1s_t = 0., [], [], []

    for i in range(epochs):

        batch_loss, n_batches = 0., 0
        
        acc_t.reset()
        f1_t.reset()
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device).float(), y.to(device)
            y_hat = model(X)
            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_t.update(y_hat, y)
            f1_t.update(y_hat, y)
            batch_loss += loss.item()
            n_batches += 1

        batch_loss = batch_loss / n_batches
        train_losses.append(batch_loss)

        f1.reset()
        model.eval()
        for X, y in val_dataloader:
            X, y = X.to(device).float(), y.to(device)
            with torch.no_grad():
                y_hat = model(X)

            f1.update(y_hat, y)

        if verbose:
            print(f"Epoch {i}:\n\ttrain loss: {batch_loss}\n\ttrain acc: {acc_t.compute().item()}\n\ttrain f1: {f1_t.compute().item()}\n\tval f1: {f1.compute().item()}")
        if best_f1 < f1.compute().item():
            best_i = i
            best_f1 = f1.compute().item()
            best_model = copy.deepcopy(model)
        
        f1s_v.append(f1.compute().item())
        f1s_t.append(f1_t.compute().item())

    return best_model, best_i, train_losses, f1s_t, f1s_v


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.

    Args:
    - seed (int): Seed value for random number generators.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def get_embeddings(model, tokenizer, inputs, device):
    inputs = ["[CLS] " + s for s in inputs]
    encoded_inputs = tokenizer(inputs, return_tensors='pt', padding=True).to(device)

    # forward pass through the model to get embeddings
    with torch.no_grad():
        output = model(**encoded_inputs)

    # extract the CLS embedding
    cls_embeddings = output.last_hidden_state[:, 0, :]
    return cls_embeddings


def add_noise(df, ratio=0.):
    """
    Adds noise to a DataFrame by changing a fraction of 'Rating' values.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        ratio (float, optional): The fraction of 'Rating' values to be modified. Default is 0.2.

    Returns:
        pandas.DataFrame: A copy of the input DataFrame with modified 'Rating' values.
    """
    # Don't change the original dataframe
    noisy_df = df.copy()
    samples = noisy_df.sample(frac=ratio)

    for i, row in samples.iterrows():
        rating = row["Rating"]
        weights = np.ones(5) * 0.25
        # We don't want the same rating, so set probability to zero
        weights[rating] = 0.
        new_rating = np.random.choice([0, 1, 2, 3, 4], 1, list(weights))[0]
        noisy_df.loc[i, "Rating"] = new_rating
    
    return noisy_df