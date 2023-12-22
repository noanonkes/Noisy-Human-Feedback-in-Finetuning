"""
Increase noise, see how it affects model performance when
using RHLF to fine tune
"""

import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

import argparse
import pandas as pd
from os.path import join as path_join

import utils
import RLHF
from dataset import SimpleDataset, RLHFRatingDataset
from model import MLP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="../data/",
                        help="Path to data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of model used to embed sentences")
    parser.add_argument("--pc", action="store_true", default=False,
                        help="Use pre-calculated embeddings")
    parser.add_argument("--ratio", type=float, default=0.,
                        help="How much noise is added to the ratings")
    parser.add_argument("--epochs-t", type=int, default=100,
                        help="Number of epochs to train for")
    parser.add_argument("--epochs-f", type=int, default=20,
                        help="Number of epochs to finetune for")
    args = parser.parse_args()

    # for reproducibility
    utils.set_seed(84)

    # use cuda if cuda is available
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    df = pd.read_csv(path_join(args.path, "tripadvisor_hotel_reviews.csv"))
    df["Rating"] -= 1 # ratings 1-5 become 0-4

    num_classes = df["Rating"].max() + 1

    train_size = int(len(df) * 0.4)
    val_size = int(len(df) * 0.1)
    finetune_size = int(len(df) * 0.4)
    test_size = len(df) - train_size - val_size - finetune_size
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    finetune_df = df.iloc[train_size+val_size:train_size+val_size+finetune_size]
    test_df = df.iloc[-test_size:]

    if args.pc:
        print("Loading embeddings...")
        embeddings = torch.load(path_join(args.path, "embeddings.pt"), map_location=device)
    else:
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        embed_model = BertModel.from_pretrained(args.model_name)
        embed_model.to(device)
        
        print("Creating embeddings...")
        embeddings = utils.get_embeddings(embed_model, tokenizer, df["Review"].to_list(), device)
        torch.save(embeddings, path_join(args.path, "embeddings.pt"))

    finetune_df = utils.add_noise(finetune_df, ratio=args.ratio)

    train_dataset = SimpleDataset(embeddings[:train_size], train_df["Rating"].to_numpy())
    val_dataset = SimpleDataset(embeddings[train_size:train_size+val_size], train_df["Rating"].to_numpy())
    test_dataset = SimpleDataset(embeddings[-test_size:], test_df["Rating"].to_numpy())
    finetune_dataset = RLHFRatingDataset(embeddings[train_size+val_size:train_size+val_size+finetune_size], finetune_df["Rating"].to_numpy())

    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Finetune dataset size:", len(finetune_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=64)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64)
    finetune_dataloader = DataLoader(finetune_dataset, shuffle=True, batch_size=64)

    # TRAIN MODEL
    print("\nTraining model...")
    model = MLP(n_in=embeddings.shape[1], layer_sizes=[768, 64, 32, 5], n_out=num_classes).to(device)
    loss_func_train = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_model, best_i, train_losses, f1s_t, f1s_v = utils.train(model, args.epochs_t, loss_func_train, optimizer, train_dataloader, val_dataloader, device, num_classes=num_classes)

    # EVALUATE INTERMEDIATE MODEL
    print(f"\nEvaluating...\nBest epoch was {best_i}")
    metrics = utils.evaluate(best_model, test_dataloader, device, num_classes=num_classes)

    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")

    # FINETUNE MODEL
    print("\nFinetuning model...")
    loss_func_finetune = RLHF.RLHFLoss(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=0.0001, weight_decay=5e-4)
    best_model, best_i, finetune_losses = RLHF.train(best_model, args.epochs_f, loss_func_finetune, optimizer, finetune_dataloader, val_dataloader, device, 1, num_classes=num_classes)

    # EVALUATE FINAL MODEL
    print("\nEvaluating...")
    print(f"\nEvaluating...\nBest epoch was {best_i}")
    metrics = utils.evaluate(best_model, test_dataloader, device, num_classes=num_classes)

    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")