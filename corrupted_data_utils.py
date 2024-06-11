from data_utils import load_data
import torch
from torch.utils.data import Dataset

# Define the path to the text file
file_path = "/home/sblotas/mober_sim/mober/geneinfo_beta.txt"

# Open the file and read its contents line by line
with open(file_path, "r") as file:
    # Initialize an empty list to store gene symbols
    landmark_gene_symbols = []

    # Iterate over each line in the file
    for line in file:
        # Split the line into fields based on tab ('\t') delimiter
        fields = line.strip().split("\t")

        # Check if the value of the feature_space column is 'landmark'
        if len(fields) > 6 and fields[6] == "landmark":
            # If yes, append the gene symbol to the list
            landmark_gene_symbols.append(
                fields[1]
            )  # Assuming gene_symbol is in the second column


def get_indices_corrupted(num_genes, must_keep_indices, missing_gene_percentage=0.1):
    # Define the number of genes to be set to -1 for augmentation
    num_missing_genes = int(
        (num_genes - len(must_keep_indices)) * missing_gene_percentage
    )

    # Generate all possible indices
    all_indices = set(range(num_genes))

    # Exclude indices that must be kept
    non_corrupted_indices = list(all_indices - set(must_keep_indices))

    # Shuffle the non-corrupted indices to ensure randomness
    shuffled_indices = torch.tensor(non_corrupted_indices)[
        torch.randperm(len(non_corrupted_indices))
    ]

    # Select the first 'num_missing_genes' shuffled indices
    indices_corrupted = shuffled_indices[:num_missing_genes]

    return indices_corrupted


class Corrupted_Dataset(Dataset):
    def __init__(self, dataframe, device, missing_gene_indices):
        self.data = dataframe
        self.device = device

        self.sources_one_hot_encoded, self.abbreviations_one_hot_encoded, values = (
            load_data(self.data, self.device)
        )

        # Set the selected gene columns to 0
        corrupted_values = values.clone()
        corrupted_values[:, missing_gene_indices] = 0
        self.corrupted_values = corrupted_values
        self.values = values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.sources_one_hot_encoded[idx],
            self.abbreviations_one_hot_encoded[idx],
            self.corrupted_values[idx],
            self.values[idx],
        )
