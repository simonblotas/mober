import pandas as pd
import re
import numpy as np
from tcga_mapper import tcga_mapper, primary, sub_type
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from typing import List, Union, Tuple, Iterator


def access_ach_data(ach_data_path: str) -> pd.DataFrame:
    """
    Access the ACH data from the given path.

    Parameters:
    ach_data_path (str): Path to the ACH data CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the ACH data.
    """
    ach_data = pd.read_csv(ach_data_path)

    # Apply the function to all column names and rename the columns
    ach_data.columns = [column.split(" ")[0] for column in ach_data.columns]
    ach_data.rename(columns={"Unnamed:": "Gene"}, inplace=True)

    # Set the index to 'Gene'
    ach_data.set_index("Gene", inplace=True)

    return ach_data


def access_tcga_data(tcga_data_path: str) -> pd.DataFrame:
    """
    Access TCGA data from the specified path.

    Parameters:
    tcga_data_path (str): Path to the TCGA data TSV file.

    Returns:
    pd.DataFrame: Transposed DataFrame of the TCGA data with 'Gene' as index.
    """
    # Read the TSV file into a pandas DataFrame
    tcga_data = pd.read_csv(tcga_data_path, sep="\t")
    tcga_data.set_index("Gene", inplace=True)

    # Transpose the DataFrame
    tcga_data_transposed = tcga_data.T

    return tcga_data_transposed


def get_intersection(ach_data: pd.DataFrame, tcga_data: pd.DataFrame) -> tuple:
    """
    Get the intersection of columns between ACH data and TCGA data.

    Parameters:
    ach_data (pd.DataFrame): DataFrame containing ACH data.
    tcga_data (pd.DataFrame): DataFrame containing TCGA data.

    Returns:
    tuple: A tuple containing two DataFrames:
           1. DataFrame containing the intersection of columns from ACH data.
           2. DataFrame containing the intersection of columns from TCGA data with reset index.
    """
    # Get the intersection of column names
    intersection_columns = ach_data.columns.intersection(tcga_data.columns)

    # Create new DataFrames with only the intersection of columns
    ach_data_intersection = ach_data[intersection_columns]
    tcga_data_intersection = tcga_data[intersection_columns]
    tcga_data_intersection.reset_index(inplace=True)

    return ach_data_intersection, tcga_data_intersection


def access_ach_data_with_abbreviation(
    ach_data_intersection: pd.DataFrame,
    tcga_projects_path: str,
    ach_metadata_file_path: str,
) -> pd.DataFrame:
    """
    Access ACH data with abbreviation.

    Parameters:
    ach_data_intersection (pd.DataFrame): DataFrame containing intersection of ACH data.
    tcga_projects_path (str): Path to the TCGA projects CSV file.
    ach_metadata_file_path (str): Path to the ACH metadata CSV file.

    Returns:
    pd.DataFrame: DataFrame containing ACH data with abbreviation.
    """
    # Read project list
    tcga_projects = pd.read_csv(tcga_projects_path)
    tcga_projects["Cancer Type"] = tcga_projects["Cancer Type"].str.lower()
    tcga_projects["Cancer Type"] = tcga_projects["Cancer Type"].replace(" or ", ",")

    # Read ACH metadata
    ach_metadata = pd.read_csv(ach_metadata_file_path)[
        ["ModelID", "OncotreePrimaryDisease", "OncotreeSubtype"]
    ]
    ach_metadata["OncotreePrimaryDisease"] = ach_metadata[
        "OncotreePrimaryDisease"
    ].str.lower()
    ach_metadata["OncotreeSubtype"] = ach_metadata["OncotreeSubtype"].str.lower()

    # Map ACH metadata to TCGA projects
    ach_metadata["tcga_project"] = ach_metadata.apply(
        lambda x: tcga_mapper(
            x["OncotreePrimaryDisease"], x["OncotreeSubtype"], primary, sub_type
        ),
        axis=1,
    )

    # Merge ACH metadata with TCGA projects
    ach_metadata_final = pd.merge(
        ach_metadata,
        tcga_projects,
        left_on="tcga_project",
        right_on="Cancer Type",
        how="left",
    ).drop("Cancer Type", axis=1)

    # Merge ACH metadata with ACH data
    ach_final = pd.merge(
        ach_metadata_final,
        ach_data_intersection,
        left_on="ModelID",
        right_on="Gene",
        how="inner",
    )
    ach_final.insert(0, "Source", "ACH")

    # Drop rows with NaN values
    ach_final = ach_final.dropna()

    return ach_final


def access_tcga_data_with_abbreviation(
    tcga_data_intersection: pd.DataFrame, tcga_metadata_file_path: str
) -> pd.DataFrame:
    """
    Access TCGA data with abbreviation.

    Parameters:
    tcga_data_intersection (pd.DataFrame): DataFrame containing intersection of TCGA data.
    tcga_metadata_file_path (str): Path to the TCGA metadata TSV file.

    Returns:
    pd.DataFrame: DataFrame containing TCGA data with abbreviation.
    """
    # Read the TSV file into a pandas DataFrame
    tcga_metadata = pd.read_csv(tcga_metadata_file_path, sep="\t")

    # Remove prefixes and suffixes from project IDs to get abbreviations
    tcga_metadata["project_id"] = [
        s.replace("TCGA-", "").replace("TARGET-", "")
        for s in tcga_metadata["project_id"]
    ]
    tcga_metadata.rename(columns={"project_id": "Abbreviation"}, inplace=True)

    # Drop duplicate rows based on 'case_submitter_id'
    tcga_metadata = tcga_metadata.drop_duplicates(
        subset="case_submitter_id", keep="first"
    )

    # Make a copy of the DataFrame
    tcga_data_intersection_copy = tcga_data_intersection.copy()

    # Modify the copy removing specified suffixes
    tcga_data_intersection_copy["index"] = [
        re.sub(r"(_S\d+|-0\d+)$", "", s) for s in tcga_data_intersection_copy["index"]
    ]

    # Merge DataFrames using 'case_submitter_id' and 'Gene'
    tcga_final = pd.merge(
        tcga_metadata,
        tcga_data_intersection_copy,
        left_on="case_submitter_id",
        right_on="index",
        how="inner",
    )

    # Insert 'Source' column
    tcga_final.insert(0, "Source", "TCGA")

    # List of valid TCGA project abbreviations
    abbreviations = [
        "LAML",
        "ACC",
        "BLCA",
        "LGG",
        "BRCA",
        "CESC",
        "CHOL",
        "LCML",
        "COAD",
        "CNTL",
        "ESCA",
        "FPPP",
        "GBM",
        "HNSC",
        "KICH",
        "KIRC",
        "KIRP",
        "LIHC",
        "LUAD",
        "LUSC",
        "DLBC",
        "MESO",
        "MISC",
        "OV",
        "PAAD",
        "PCPG",
        "PRAD",
        "READ",
        "SARC",
        "SKCM",
        "STAD",
        "TGCT",
        "THYM",
        "THCA",
        "UCS",
        "UCEC",
        "UVM",
    ]

    # Create a boolean mask to filter rows based on whether the abbreviation is valid
    mask = tcga_final["Abbreviation"].isin(abbreviations)

    # Apply the mask to filter the DataFrame
    tcga_final = tcga_final[mask]

    return tcga_final


def abbreviation_to_one_hot(
    abbreviation: str, abbreviation_to_index: dict
) -> torch.Tensor:
    """
    Map an abbreviation to its one-hot encoded tensor.

    Parameters:
    - abbreviation (str): The abbreviation to be encoded.
    - abbreviation_to_index (dict): A dictionary mapping abbreviations to their corresponding indices.

    Returns:
    - one_hot_tensor (torch.Tensor): The one-hot encoded tensor for the given abbreviation.
    """
    if " or " in abbreviation:
        # Handle cases where an abbreviation may have two classes
        classes = abbreviation.split(" or ")
        indices = [abbreviation_to_index[c] for c in classes]
        num_classes = len(abbreviation_to_index)
        one_hot_tensor = torch.zeros(num_classes)
        one_hot_tensor[indices] = 1
        return one_hot_tensor
    elif abbreviation in abbreviation_to_index:
        # Handle cases where an abbreviation has a single class
        index = abbreviation_to_index[abbreviation]
        num_classes = len(abbreviation_to_index)
        one_hot_tensor = F.one_hot(
            torch.tensor(index), num_classes=num_classes
        ).squeeze()
        return one_hot_tensor
    else:
        raise ValueError(
            f"Abbreviation '{abbreviation}' not found in the provided mapping."
        )


abbreviation_to_index = {
    "LAML": 0,
    "ACC": 1,
    "BLCA": 2,
    "LGG": 3,
    "BRCA": 4,
    "CESC": 5,
    "CHOL": 6,
    "LCML": 7,
    "COAD": 8,
    "CNTL": 9,
    "ESCA": 10,
    "FPPP": 11,
    "GBM": 12,
    "HNSC": 13,
    "KICH": 14,
    "KIRC": 15,
    "KIRP": 16,
    "LIHC": 17,
    "LUAD": 18,
    "LUSC": 19,
    "DLBC": 20,
    "MESO": 21,
    "MISC": 22,
    "OV": 23,
    "PAAD": 24,
    "PCPG": 25,
    "PRAD": 26,
    "READ": 27,
    "SARC": 28,
    "SKCM": 29,
    "STAD": 30,
    "TGCT": 31,
    "THYM": 32,
    "THCA": 33,
    "UCS": 34,
    "UCEC": 35,
    "UVM": 36,
}

# Reverse the dictionary
index_to_abbreviation = {v: k for k, v in abbreviation_to_index.items()}


def one_hot_to_abbreviation(
    one_hot_tensor: torch.Tensor, index_to_abbreviation: dict
) -> Union[str, List[str]]:
    """
    Map a one-hot encoded tensor to its corresponding abbreviation.

    Parameters:
    - one_hot_tensor (torch.Tensor): The one-hot encoded tensor to be decoded.
    - index_to_abbreviation (dict): A dictionary mapping indices to their corresponding abbreviations.

    Returns:
    - abbreviation (str or List[str]): The abbreviation for the given one-hot encoded tensor.
    """

    non_zero_indices = torch.nonzero(one_hot_tensor).squeeze()

    if non_zero_indices.numel() == 1:
        # If there is only one non-zero element in the tensor
        index = non_zero_indices.item()
        if index in index_to_abbreviation:
            abbreviation = index_to_abbreviation[index]
            return abbreviation
        else:
            raise ValueError(
                f"Index '{index}' not found in the provided reverse mapping."
            )
    elif non_zero_indices.numel() == 2:
        # If there are two non-zero elements in the tensor (representing a combination of classes)
        abbreviations = [index_to_abbreviation[idx.item()] for idx in non_zero_indices]
        abr = ""
        for i in range(len(abbreviations) - 1):
            abr += abbreviations[i] + " or "
        abr += abbreviations[len(abbreviations) - 1]
        return abr
    else:
        # If the tensor does not have exactly one or two non-zero elements
        raise ValueError(
            "Input tensor is not a valid one-hot encoded tensor (should have exactly one or two non-zero elements)."
        )


def source_to_one_hot(source: str, source_to_index: dict) -> torch.Tensor:
    """
    Map a source to its one-hot encoded tensor.

    Parameters:
    - source (str): The source to be encoded ('ACH' or 'TCGA').
    - source_to_index (dict): A dictionary mapping sources to their corresponding indices.

    Returns:
    - one_hot_tensor (torch.Tensor): The one-hot encoded tensor for the given source.
    """

    if source in source_to_index:
        index = source_to_index[source]
        num_classes = len(source_to_index)
        one_hot_tensor = torch.zeros(num_classes)
        one_hot_tensor[index] = 1
        return one_hot_tensor
    else:
        raise ValueError(f"Source '{source}' not found in the provided mapping.")


def one_hot_to_source(one_hot_tensor: torch.Tensor, index_to_source: dict) -> str:
    """
    Map a one-hot encoded tensor to its corresponding source.

    Parameters:
    - one_hot_tensor (torch.Tensor): The one-hot encoded tensor to be decoded.
    - index_to_source (dict): A dictionary mapping indices to their corresponding sources.

    Returns:
    - source (str): The source for the given one-hot encoded tensor.
    """

    if torch.sum(one_hot_tensor) == 1:
        index = torch.argmax(one_hot_tensor).item()
        if index in index_to_source:
            source = index_to_source[index]
            return source
        else:
            raise ValueError(
                f"Index '{index}' not found in the provided reverse mapping."
            )
    else:
        raise ValueError(
            "Input tensor is not a valid one-hot encoded tensor (should have exactly one non-zero element)."
        )


source_to_index = {"ACH": 0, "TCGA": 1}
index_to_source = {0: "ACH", 1: "TCGA"}


def load_data(
    data: pd.DataFrame, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load data from a pandas DataFrame and convert it to PyTorch tensors.

    Parameters:
    - data: The DataFrame containing the data.
    - device: The device to store the tensors on.

    Returns:
    - A tuple containing three tensors:
      1. sources_one_hot_encoded: The one-hot encoded tensor for the data sources.
      2. abbreviations_one_hot_encoded: The one-hot encoded tensor for the data abbreviations.
      3. values: The tensor containing the numerical values.
    """
    # Extract numerical values from the DataFrame
    values = data.iloc[:, -18178:].values
    # Convert numerical values to a PyTorch tensor of type float32
    values = torch.tensor(values).to(torch.float32)

    # Extract metadata (abbreviations and sources) from the DataFrame
    metadata = data.iloc[:, :-18178]
    abbreviations = metadata["Abbreviation"]
    # Convert abbreviations to one-hot encoded tensors using the provided function
    abbreviations_one_hot_encoded = [
        abbreviation_to_one_hot(abbreviation, abbreviation_to_index)
        for abbreviation in abbreviations
    ]
    # Stack the one-hot encoded tensors along the specified dimension
    abbreviations_one_hot_encoded = torch.stack(abbreviations_one_hot_encoded, dim=0)

    sources = metadata["Source"]
    # Convert sources to one-hot encoded tensors using the provided function
    sources_one_hot_encoded = [
        source_to_one_hot(source, source_to_index) for source in sources
    ]
    # Stack the one-hot encoded tensors along the specified dimension
    sources_one_hot_encoded = torch.stack(sources_one_hot_encoded, dim=0)

    # Move tensors to the specified device
    values.to(device)
    abbreviations_one_hot_encoded.to(device)
    sources_one_hot_encoded.to(device)

    return sources_one_hot_encoded, abbreviations_one_hot_encoded, values


class CustomDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, device: torch.device):
        """
        Initialize the CustomDataset.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the data.
        - device (torch.device): The device where tensors will be moved.
        """
        self.data = dataframe
        self.device = device
        (
            self.sources_one_hot_encoded,
            self.abbreviations_one_hot_encoded,
            self.values,
        ) = load_data(self.data, self.device)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
        - int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing tensors for sources,
          abbreviations, and values corresponding to the specified index.
        """
        return (
            self.sources_one_hot_encoded[idx],
            self.abbreviations_one_hot_encoded[idx],
            self.values[idx],
        )


def access_data(
    ach_data_path: str,
    tcga_projects_path: str,
    ach_metadata_file_path: str,
    tcga_data_path: str,
    tcga_metadata_file_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Access and process the ACH and TCGA data.

    Parameters:
    - ach_data_path (str): The path to the ACH data.
    - tcga_projects_path (str): The path to the TCGA projects.
    - ach_metadata_file_path (str): The path to the ACH metadata file.
    - tcga_data_path (str): The path to the TCGA data.
    - tcga_metadata_file_path (str): The path to the TCGA metadata file.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing DataFrames for the processed ACH
      and TCGA data.
    """
    ach_data = access_ach_data(ach_data_path)
    tcga_data = access_tcga_data(tcga_data_path)
    ach_data_intersection, tcga_data_intersection = get_intersection(
        ach_data, tcga_data
    )
    ach_final = access_ach_data_with_abbreviation(
        ach_data_intersection, tcga_projects_path, ach_metadata_file_path
    )
    tcga_final = access_tcga_data_with_abbreviation(
        tcga_data_intersection, tcga_metadata_file_path
    )
    return ach_final, tcga_final


class StratifiedSampler(Sampler):
    def __init__(self, dataset: Dataset, indices: List[int]):
        """
        Stratified sampling implementation for creating a balanced sampler.

        Parameters:
        - dataset (Dataset): The dataset to be sampled from.
        - indices (List[int]): The indices to be sampled from.

        Note:
        The dataset is expected to return a tuple (label, _, _) from its __getitem__ method,
        where label is the class label of the sample.
        """
        self.dataset = dataset
        self.indices = indices
        self.num_samples = len(indices)

        # Count the frequency of each class in the specified indices
        label_to_count = {}
        for index in self.indices:
            label, _, _ = self.dataset[index]
            label = str(label.tolist())
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # Weight for each sample
        weights = [
            1.0 / label_to_count[str(self.dataset[index][0].tolist())]
            for index in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the indices.

        Returns:
        - Iterator[int]: An iterator over the indices.
        """
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in the sampler.

        Returns:
        - int: The number of samples in the sampler.
        """
        return self.num_samples
