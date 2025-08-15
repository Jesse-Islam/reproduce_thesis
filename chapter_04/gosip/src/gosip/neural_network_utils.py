import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


def calculate_category_accuracy(predicted_logits, goal_categories, num_categories):
    """
    calculates categorical accuracy and the category specific distances from goal values.
    Parameters:
    - predicted_logits
    - goal_categories: where the soft_max of logits should be.
    - num_categories: number of categories in each categorical variable.
    Returns:
    - accuracies: the overall accuracy of the prediction.
    - distances: distances for each categorical variable we track.
    """
    start_idx = 0
    accuracies = []
    distances = []
    for _, size in enumerate(num_categories):
        end_idx = start_idx + size
        # Apply softmax activation to convert logits to probabilities
        predicted_probs = functional.softmax(predicted_logits[:, start_idx:end_idx], dim=1)
        distances.append((goal_categories[:, start_idx:end_idx] - predicted_probs).abs().mean(dim=1).cpu().detach())
        # Get the predicted labels by taking the argmax along the category dimension
        predicted_labels = torch.argmax(predicted_probs, dim=1)
        # Get the goal labels by taking the argmax along the same dimension
        goal_labels = torch.argmax(goal_categories[:, start_idx:end_idx], dim=1)
        # Calculate accuracy for current segment of categories
        correct = (predicted_labels == goal_labels).sum().item()
        total = predicted_labels.size(0)
        accuracy = correct / total
        accuracies.append(accuracy)
        start_idx = end_idx

    return accuracies, distances

def calculate_category_total_accuracy(predicted_logits: torch.Tensor,
                                      goal_categories:  torch.Tensor,
                                      num_categories:   list[int]):
    if sum(num_categories) != predicted_logits.shape[1]:
        raise ValueError("sum(num_categories) must equal logits width")
    if predicted_logits.shape != goal_categories.shape:
        raise ValueError("predicted_logits and goal_categories must have identical shape")
    if predicted_logits.device != goal_categories.device or predicted_logits.dtype != goal_categories.dtype:
        goal_categories = goal_categories.to(predicted_logits.device, predicted_logits.dtype)

    logit_blocks = torch.split(predicted_logits, num_categories, dim=1)
    goal_blocks  = torch.split(goal_categories,  num_categories, dim=1)
    accuracies      = []
    total_distance  = torch.zeros(predicted_logits.size(0),
                                  device=predicted_logits.device,
                                  dtype=predicted_logits.dtype)

    for logits_blk, goal_blk in zip(logit_blocks, goal_blocks):
        probs = F.softmax(logits_blk, dim=1)
        total_distance += (goal_blk - probs).abs().sum(dim=1)
        acc = (probs.argmax(dim=1) == goal_blk.argmax(dim=1)).float().mean().item()
        accuracies.append(acc)
    #print(total_distance)
    return accuracies, [total_distance]


def calculate_category_loss(ce_loss, predicted_categories, goal_categories, num_categories, weights_all):
    """
    calculates categorical loss.
    Parameters:
    - ce_loss: function for cross entropy loss. user could specify another given it makes sense.
    - predicted_logits: 
    - goal_categories: where the soft_max of logits should be.
    - num_categories: number of categories in each categorical variable.
    - weights_all: each sample has a weight for its particular category. 
      This weight is passed in to upweight under-represented categories
    Returns:
    - weighted_categorical_loss 
    """
    start_idx = 0
    cat_loss = 0
    for _, size in enumerate(num_categories):
        end_idx = start_idx + size
        # Apply weights specific to each category slice
        # weights = weights_all[:,i]
        # Calculate loss for current segment of categories
        # print(predicted_categories[:, start_idx:end_idx], goal_categories[:, start_idx:end_idx] , weights_all[:,i])
        label_loss = ce_loss(
            predicted_categories[:, start_idx:end_idx], goal_categories[:, start_idx:end_idx], weights_all
        )
        cat_loss += label_loss.sum()  # Sum up the weighted loss for current category set
        start_idx = end_idx
    return cat_loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, predictions, targets, weights):
        # Compute cross-entropy loss without averaging
        losses = self.ce_loss(predictions, targets)
        # Apply weights
        weighted_losses = losses * weights
        # Return the mean loss
        return weighted_losses.mean()





class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, targets, weights):
        # Compute BCE with logits loss without reduction
        losses = self.bce_loss(predictions, targets)
        # Apply weights
        weighted_losses = losses * weights
        # Return the mean loss
        return weighted_losses.mean()


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(self, predictions, targets, weights):
        # Compute L1 loss without reduction
        losses = self.l1_loss(predictions, targets)
        # Apply weights to each loss element
        weighted_losses = losses * weights
        # Return the mean of these weighted losses
        return weighted_losses.mean()


class AnnDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor, weights_tensor, global_weights):
        self.data = data_tensor
        self.labels = labels_tensor
        self.weights = weights_tensor
        self.global_weights = global_weights
        # self.classes = classes

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.weights[idx], self.global_weights[idx]



def calculate_inverse_frequency_weights(tensors, num_categories):
    """
    calculates inverse frequency weights for each categorical variable
    Parameters:
    - tensors: the transition label tensors.
    - num_categories: number of categories for each categorical variable.
    returns: the normalized weights aligned to each sample.
    """
    output_vectors = []
    start_idx = 0
    for _, size in enumerate(num_categories):
        end_idx = start_idx + size  # Compute ending index for current label
        # Calculate loss for the current label's output slice
        tensor = tensors[:, start_idx:end_idx]
        unique_rows, inverse_indices = torch.unique(tensor, dim=0, return_inverse=True)
        # Count occurrences of each unique row
        counts = torch.bincount(inverse_indices, minlength=unique_rows.size(0))
        # Use the inverse of the counts, and then normalize to prevent extremely large weights
        inverse_frequencies = 1.0 / counts.float()
        # Normalize the weights to make sure their sum equals the number of samples
        normalized_weights = inverse_frequencies / (inverse_frequencies.sum() * tensor.size(0))
        # Map normalized weights back to the original tensor using inverse indices
        output_vectors.append(normalized_weights[inverse_indices].view(-1, 1))
        start_idx = end_idx
    return torch.cat(output_vectors, dim=1)



def calculate_global_inverse_frequency_weights(tensors):
    """
    calculates inverse frequency weights for the entire label profile.
    Parameters:
    - tensors: the transition label tensors.
    - num_categories: number of categories for each categorical variable.
    returns: the normalized weights aligned to each sample for their entire label profile.
    """
    unique_rows, inverse_indices = torch.unique(tensors, dim=0, return_inverse=True)
    counts = torch.bincount(inverse_indices, minlength=unique_rows.size(0))
    inverse_frequencies = 1.0 / counts.float()
    normalized_weights = inverse_frequencies / (inverse_frequencies.sum() )
    global_weights = normalized_weights[inverse_indices]
    global_weights = global_weights/global_weights.min()
    return global_weights.view(-1, 1)

def prepare_data(adata, num_categories, batch_size=16):
    """
    prepare the dataset into a dataloader.
    Parameter:
    - adata: adata object.
    - num_categories: number of categories for each categorical variable.
    - batch_size: batch size to be used in each iteration of an epoch (16)
    return: dataloader with data, labels and weights.
    """
    # adata is anndata format,assumes uncompressed
    labels_tensor = torch.tensor(adata.one_hot_labels.values.copy(), dtype=torch.float32)
    data_tensor = torch.tensor(adata.X.copy(), dtype=torch.float32)
    weight_tensors = torch.tensor(
        calculate_inverse_frequency_weights(labels_tensor, num_categories).detach(), dtype=torch.float32
    )
    global_weights = torch.tensor(
        calculate_global_inverse_frequency_weights(labels_tensor).detach(), dtype=torch.float32
    )
    dataset = AnnDataset(data_tensor, labels_tensor, weight_tensors, global_weights)
    # print(weight_tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def calculate_category_total_accuracy_old(predicted_logits, goal_categories, num_categories):
    """
    calculates categorical accuracy and the category specific distances (which are averaged) from goal values.
    Parameters:
    - predicted_logits
    - goal_categories: where the soft_max of logits should be.
    - num_categories: number of categories in each categorical variable.
    Returns:
    - accuracies: the overall accuracy of the prediction.
    - [total_distance]: distances for each categorical variable we track are averaged.
    """
    start_idx = 0
    accuracies = []
    distances = []
    total_distance=torch.zeros(predicted_logits.shape[0])
    for _, size in enumerate(num_categories):
        end_idx = start_idx + size
        # Apply softmax activation to convert logits to probabilities
        predicted_probs = functional.softmax(predicted_logits[:, start_idx:end_idx], dim=1)
        distance = (goal_categories[:, start_idx:end_idx] - predicted_probs).abs().sum(dim=1).cpu().detach()
        distances.append((goal_categories[:, start_idx:end_idx] - predicted_probs).abs().sum(dim=1).cpu().detach())
        total_distance += distance
        # Get the predicted labels by taking the argmax along the category dimension
        predicted_labels = torch.argmax(predicted_probs, dim=1)
        # Get the goal labels by taking the argmax along the same dimension
        goal_labels = torch.argmax(goal_categories[:, start_idx:end_idx], dim=1)
        # Calculate accuracy for current segment of categories
        correct = (predicted_labels == goal_labels).sum().item()
        total = predicted_labels.size(0)
        accuracy = correct / total
        accuracies.append(accuracy)
        start_idx = end_idx
    return accuracies, [total_distance]
