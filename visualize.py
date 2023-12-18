import torch

import argparse
import pandas as pd
from os.path import join as path_join

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="data/",
                        help="Path to data")
    args = parser.parse_args()

    # use cuda if cuda is available
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    df = pd.read_csv("data/tripadvisor_hotel_reviews.csv")

    print("Loading embeddings...")
    embeddings = torch.load(path_join(args.path, "embeddings.pt"), map_location=device)

    print("Reducing the embeddings...")
    reducer = umap.UMAP()
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    s=5,
    alpha=0.7,
    c=[sns.color_palette()[x] for x in df.Rating.to_list()])

    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the hotel reviews dataset', fontsize=24)
    plt.legend()
    plt.show()

