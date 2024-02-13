import torch
import numpy as np
from torch.utils.data import DataLoader
import faiss


def project_into_decoded_space(
    dataloader, model_BatchAE, device, space_to_project_into="TCGA"
):
    all_decoded_data = []
    true_abbreviations = []
    ach_decoded_data = []
    ach_true_abbreviations = []
    tcga_decoded_data = []
    tcga_true_abbreviations = []
    model_BatchAE.to(device)
    model_BatchAE.eval()
    for batch in dataloader:
        genes, labels, abbreviations = (
            batch[2].to(device),
            batch[0].to(device),
            batch[1].to(device),
        )
        means, stdev, encoded_data = model_BatchAE.encoder(genes)
        # Create tensors with ones and zeros
        n_rows = labels.shape[0]
        ones_column = torch.ones(n_rows, 1, device=device)
        zeros_column = torch.zeros(n_rows, 1, device=device)
        mask_is_in_tcga = torch.all(
            labels == torch.tensor([0.0, 1.0]).to(device), dim=1
        )
        mask_is_in_ach = torch.all(labels == torch.tensor([1.0, 0.0]).to(device), dim=1)

        # Concatenate tensors along the second dimension (dim=1)
        if space_to_project_into == "ACH":
            new_labels = torch.cat((ones_column, zeros_column), dim=1)
        elif space_to_project_into == "TCGA":
            new_labels = torch.cat((zeros_column, ones_column), dim=1)

        # Ensure that new_labels is moved to the same device as encoded_data
        new_labels = new_labels.to(device)

        decoded_data = model_BatchAE.decoder(encoded_data, new_labels)
        # Append the decoded data to the list
        all_decoded_data.append(
            decoded_data.cpu().detach().numpy()
        )  # Assuming you want to collect the results
        true_abbreviations.append(abbreviations.cpu().detach().numpy())
        ach_decoded_data.append(decoded_data[mask_is_in_ach].cpu().detach().numpy())
        ach_true_abbreviations.append(
            abbreviations[mask_is_in_ach].cpu().detach().numpy()
        )
        tcga_decoded_data.append(decoded_data[mask_is_in_tcga].cpu().detach().numpy())
        tcga_true_abbreviations.append(
            abbreviations[mask_is_in_tcga].cpu().detach().numpy()
        )

    # Concatenate decoded data from all batches
    all_decoded_data = np.concatenate(all_decoded_data, axis=0)
    true_abbreviations = np.concatenate(true_abbreviations, axis=0)
    ach_decoded_data = np.concatenate(ach_decoded_data, axis=0)
    ach_true_abbreviations = np.concatenate(ach_true_abbreviations, axis=0)
    tcga_decoded_data = np.concatenate(tcga_decoded_data, axis=0)
    tcga_true_abbreviations = np.concatenate(tcga_true_abbreviations, axis=0)

    return (
        all_decoded_data,
        true_abbreviations,
        ach_decoded_data,
        ach_true_abbreviations,
        tcga_decoded_data,
        tcga_true_abbreviations,
    )


def metrics_on_dataloader(
    true_space_labels, true_space_features, true_labels, model_predicted_features, k=25
):

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

    # Convert one-hot representation to abbreviation
    # predicted_abbreviations = one_hot_to_abbreviation(torch.tensor(new_array), index_to_abbreviation)

    # Count correct predictions
    count = np.sum(np.all((true_labels) == (new_array), axis=1)).item()

    return count / len(true_labels)