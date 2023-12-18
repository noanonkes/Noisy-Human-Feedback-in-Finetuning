import torch
import numpy as np

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
        new_rating = np.random.choice([1, 2, 3, 4, 5], 1, list(weights))[0]
        noisy_df.loc[i, "Rating"] = new_rating
    
    return noisy_df