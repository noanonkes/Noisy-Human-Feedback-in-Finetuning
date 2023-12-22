import torch
from torch import nn
import torch.nn.functional as F
import copy
from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy


class RLHFLoss(nn.Module):
    def __init__(self, device, k=1):
        """
        Initializes RLHFLoss.

        Args:
        - device (str): The device to perform computations on.
        - k (int, optional): The number of comparisons to make (default is 1).
        """
        super(RLHFLoss, self).__init__()
        self.device = device
        self.k = k

    def forward(self, ratings_hat, ratings):
        """
        Computes the loss based on predicted ratings pair and true rating pairs.

        Args:
        - ratings_hat (torch.Tensor): Predicted ratings tensor of shape (B, k + 1, 5).
        - ratings (torch.Tensor): True ratings tensor of shape (B, k + 1).
          The ratings are expected to be in descending order.

        Returns:
        - torch.Tensor: Computed loss tensor.

        Note:
        - B represents the batch size.
        """
        ratings = torch.cat((self.k + 1) * [ratings[:,0].unsqueeze(-1)], dim=-1)

        # Extract reward for each rating pair (R+ and R-) using gathered indices
        reward_per_rating = torch.gather(F.softplus(ratings_hat, beta=0.1), 2, ratings.unsqueeze(-1)).squeeze(-1)

        # Compute R+ - R-
        reward_diff = torch.stack([reward_per_rating[:, i] - reward_per_rating[:, i+1] for i in range(reward_per_rating.shape[-1]-1)], dim=1)

        # Compute loss as log(sigmoid(R+ - R-))
        loss = F.logsigmoid(reward_diff)

        # Sum up the losses for each pair (R+ - R-) and return the negative sum as the final loss
        return -torch.sum(loss)
    

        
def train(model, epochs, loss_func, optimizer, t_dataloader, v_dataloader, device, k, num_classes=4, verbose=False):
    """
    Trains a given model using the specified parameters.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        epochs (int): The number of training epochs.
        loss_func (callable): The loss function used for optimization.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        dataloader (torch.utils.data.DataLoader): DataLoader for iterating through the dataset.
        device (torch.device): The device (CPU/GPU) to be used for computation.
        k (int): The value 'k' representing a parameter in the model.
        num_classes (int, optional): The number of output classes. Default is 5.

    Returns:
        torch.nn.Module: The best trained model.
        list: List containing losses at each epoch during training.
    """
    # Initialize evaluation metrics
    f1 = MulticlassF1Score(num_classes=num_classes)
    f1_t = MulticlassF1Score(num_classes=num_classes)
    acc_t = MulticlassAccuracy(num_classes=num_classes)

    # Initialize variables for tracking best loss and storing losses at each epoch
    best_f1, losses, f1s_v, f1s_t = 0., [], [], []

    # Loop through epochs
    for i in range(epochs):

        # Initialize variables for tracking batch loss and number of batches
        batch_loss, n_batches = 0., 0

        # Reset train accuracy for this epoch
        acc_t.reset()
        f1_t.reset()

        # Set the model in training mode
        model.train()
        for m_X, m_y in t_dataloader:
            m_X, m_y = m_X.to(device), m_y.to(device)

            batch_size = len(m_X)

            # Reshape the input embeddings for model: B x k + 1 x 768 -> B * (k + 1) x 768
            m_X = m_X.view(-1, m_X.shape[-1]).to(device)

            # Forward pass: compute predictions: B * (k + 1) x 5
            m_y_hat = model(m_X.float())

            # Update evaluation metrics
            acc_t.update(m_y_hat, m_y.view(-1))
            f1_t.update(m_y_hat, m_y.view(-1))

            # Reshape predictions to original shape: B * (k + 1) x 768 -> B x k + 1 x 768
            m_y_hat = m_y_hat.view((batch_size, k + 1, num_classes))
            
            # Calculate loss
            loss = loss_func(m_y_hat, m_y)

            # Perform backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch loss and update number of batches processed
            batch_loss += loss.item()
            n_batches += 1

        # Calculate average batch loss
        batch_loss = batch_loss / n_batches
        losses.append(batch_loss)

        # Reset evaluation metrics for each epoch
        f1.reset()
        model.eval()
        for X, y in v_dataloader:
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
    return best_model, best_i, losses, f1s_t, f1s_v