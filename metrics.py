import torch
import numpy as np
from torch.utils.data import DataLoader
import faiss


def project_into_decoded_space(
    dataloader, model_BatchAE, device, space_to_project_into="TCGA"
):
    """
    Project the data from the latent space into the decoded space using the specified model.

    Parameters:
    - dataloader (DataLoader): The DataLoader containing the data batches.
    - model_BatchAE: The trained BatchAE model.
    - device: The device to use for computations.
    - space_to_project_into (str): The space to project the data into, either "CCLE" or "TCGA".

    Returns:
    - A tuple containing decoded data and true abbreviations for all, CCLE, and TCGA spaces.
    """
    all_decoded_data = []
    true_abbreviations = []
    ccle_decoded_data = []
    ccle_true_abbreviations = []
    tcga_decoded_data = []
    tcga_true_abbreviations = []

    # Move model to the specified device and set it to evaluation mode
    model_BatchAE.to(device)
    model_BatchAE.eval()

    for batch in dataloader:
        genes, labels, abbreviations = (
            batch[2].to(device),
            batch[0].to(device),
            batch[1].to(device),
        )
        # Encode the input data
        means, stdev, encoded_data = model_BatchAE.encoder(genes)

        # Create tensors for masking
        n_rows = labels.shape[0]
        ones_column = torch.ones(n_rows, 1, device=device)
        zeros_column = torch.zeros(n_rows, 1, device=device)
        mask_is_in_tcga = torch.all(
            labels == torch.tensor([0.0, 1.0]).to(device), dim=1
        )
        mask_is_in_ccle = torch.all(labels == torch.tensor([1.0, 0.0]).to(device), dim=1)

        # Concatenate tensors along the second dimension (dim=1) based on the space to project into
        if space_to_project_into == "CCLE":
            new_labels = torch.cat((ones_column, zeros_column), dim=1)
        elif space_to_project_into == "TCGA":
            new_labels = torch.cat((zeros_column, ones_column), dim=1)

        # Move new_labels to the same device as encoded_data
        new_labels = new_labels.to(device)

        # Decode the encoded data with the new labels
        decoded_data = model_BatchAE.decoder(encoded_data, new_labels)

        # Append decoded data and true abbreviations based on the masks
        all_decoded_data.append(decoded_data.cpu().detach().numpy())
        true_abbreviations.append(abbreviations.cpu().detach().numpy())
        ccle_decoded_data.append(decoded_data[mask_is_in_ccle].cpu().detach().numpy())
        ccle_true_abbreviations.append(
            abbreviations[mask_is_in_ccle].cpu().detach().numpy()
        )
        tcga_decoded_data.append(decoded_data[mask_is_in_tcga].cpu().detach().numpy())
        tcga_true_abbreviations.append(
            abbreviations[mask_is_in_tcga].cpu().detach().numpy()
        )

    # Concatenate decoded data and true abbreviations from all batches
    all_decoded_data = np.concatenate(all_decoded_data, axis=0)
    true_abbreviations = np.concatenate(true_abbreviations, axis=0)
    ccle_decoded_data = np.concatenate(ccle_decoded_data, axis=0)
    ccle_true_abbreviations = np.concatenate(ccle_true_abbreviations, axis=0)
    tcga_decoded_data = np.concatenate(tcga_decoded_data, axis=0)
    tcga_true_abbreviations = np.concatenate(tcga_true_abbreviations, axis=0)

    return (
        all_decoded_data,
        true_abbreviations,
        ccle_decoded_data,
        ccle_true_abbreviations,
        tcga_decoded_data,
        tcga_true_abbreviations,
    )


def metrics_on_dataloader(
    true_space_labels: np.ndarray,
    true_space_features: np.ndarray,
    true_labels: np.ndarray,
    model_predicted_features: np.ndarray,
    k: int = 25,
) -> float:
    """
    Calculate metrics on the data loader.

    Parameters:
    - true_space_labels (np.ndarray): True space labels.
    - true_space_features (np.ndarray): True space features.
    - true_labels (np.ndarray): True labels.
    - model_predicted_features (np.ndarray): Predicted features from the model.
    - k (int, optional): Number of nearest neighbors to consider. Defaults to 25.

    Returns:
    - float: Accuracy metric.
    """

    faiss.normalize_L2(true_space_features)
    index = faiss.IndexFlatIP(true_space_features.shape[1])
    index.add(true_space_features)

    # Perform nearest neighbor search for all elements in model_predicted_features
    distances, indices = index.search(model_predicted_features, k=k)

    # Get the labels of the nearest neighbors
    nearest_neighbors_labels = true_space_labels[indices]

    # Calculate the average class of the nearest neighbors for each element
    average_class = np.sum(nearest_neighbors_labels, axis=1)

    # Find the index of the maximum value
    max_indices = np.argmax(average_class, axis=1)

    # Create a new array with zeros
    new_array = np.zeros_like(average_class)

    # Set the value at the max_index to 1 for each element
    new_array[np.arange(len(max_indices)), max_indices] = 1

    # Count correct predictions
    count = np.sum(np.all((true_labels) == (new_array), axis=1)).item()

    return count / len(true_labels)
