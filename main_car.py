"""
Increase noise, see how it affects model performance when
using RHLF to fine tune
"""

import torch, argparse
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from os.path import join as path_join

import utils
import RLHF
from dataset import RLHFCarDataset, SimpleDataset
from model import SimpleMLP, SimplerMLP

from train import train
from evaluate import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/",
                        help="Path to data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of model used to embed sentences")
    parser.add_argument("--pc", action="store_true", default=False,
                        help="Use pre-calculated embeddings")
    parser.add_argument("--ratio", type=float, default=0.,
                        help="How much noise is added to the ratings")
    parser.add_argument("--k", type=int, default=1,
                        help="How many comparisons to use when doing RLHF")
    parser.add_argument("--epochs-t", type=int, default=100,
                        help="Number of epochs to train for")
    parser.add_argument("--epochs-f", type=int, default=20,
                        help="Number of epochs to finetune for")
    args = parser.parse_args()

    # for reproducibility
    utils.set_seed(42)

    # use cuda if cuda is available
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    df = pd.read_csv("data/car+evaluation/car.data", names=["buying", "maint",
                                                              "doors", "persons",
                                                              "lug_boot", "safety",
                                                              "acc"])
    
    # Example using One-Hot Encoding
    X = []
    for col in df.columns:
        if col == "acc":
            x = df[col].to_numpy()
            x[x == "unacc"] = 0
            x[x == "acc"] = 1
            x[x == "good"] = 2
            x[x == "vgood"] = 3

            X.append(x.astype(int))
            continue

        one_hot_encoder = OneHotEncoder()
        x = one_hot_encoder.fit_transform(df[[col]]).toarray()
        X.append(x)
        
    labels = X[-1].astype(int)
    X = np.concatenate(X[:-1], axis=-1)

    full_dataset = SimpleDataset(X, labels)
    num_classes = int(np.max(labels) + 1) # int is necessary otherwise error

    train_size = int(len(df) * 0.2)
    val_size = int(len(df) * 0.1)
    finetune_size = int(len(df) * 0.6)
    test_size = len(df) - train_size - val_size - finetune_size

    idx = np.arange(len(full_dataset))
    np.random.shuffle(idx)

    train_dataset = SimpleDataset(X[idx[:train_size]], labels[idx[:train_size]])
    val_dataset = SimpleDataset(X[idx[train_size:train_size+val_size]], labels[idx[train_size:train_size+val_size]])
    test_dataset = SimpleDataset(X[idx[-test_size:]], labels[idx[-test_size:]])
    finetune_dataset = RLHFCarDataset(X[idx[train_size+val_size:train_size+val_size+finetune_size]], labels[idx[train_size+val_size:train_size+val_size+finetune_size]], args.k)

    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Finetune dataset size:", len(finetune_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=32)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32)
    finetune_dataloader = DataLoader(finetune_dataset, shuffle=True, batch_size=32)

    # # TRAIN MODEL
    print("\nTraining model...")
    model = SimpleMLP(n_in=X.shape[1], n_out=4).to(device)
    loss_func_train = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_model, train_losses = train(model, args.epochs_t, loss_func_train, optimizer, train_dataloader, val_dataloader, device, num_classes=num_classes)

    # EVALUATE INTERMEDIATE MODEL
    print("\nEvaluating...")
    metrics = evaluate(best_model, test_dataloader, device, num_classes=num_classes)

    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")

    # FINETUNE MODEL
    print("\nFinetuning model...")
    loss_func_finetune = RLHF.RLHFLoss(device, args.k)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=0.0001, weight_decay=5e-4)
    best_model, finetune_losses = RLHF.train(best_model, args.epochs_f, loss_func_finetune, optimizer, finetune_dataloader, val_dataloader, device, args.k, num_classes=num_classes)

    # EVALUATE FINAL MODEL
    print("\nEvaluating...")
    metrics = evaluate(best_model, test_dataloader, device, num_classes=num_classes)

    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")