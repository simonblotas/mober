from sklearn.base import BaseEstimator
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from models import BatchVAE, MLP
from data_utils import StratifiedSampler, CustomDataset
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    BatchSampler,
    RandomSampler,
    SubsetRandomSampler,
    Sampler,
)
from torch.utils.data import Dataset
from torch import optim
from sklearn.model_selection import train_test_split
from loss import loss_function_vae, loss_function_classification
from corrupted_data_utils import (
    get_indices_corrupted,
    landmark_gene_symbols,
    Corrupted_Dataset,
)
from data_utils import load_data, access_data, load_data_lite
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, features, device, sources_one_hot_encoded):
        self.data = features.to(device)
        self.device = device
        self.sources_one_hot_encoded = sources_one_hot_encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx], self.sources_one_hot_encoded


class Corrupted_Dataset(Dataset):
    def __init__(self, features, device, sources_one_hot_encoded, missing_gene_indices):
        self.data = features.to(device)
        self.device = device
        self.sources_one_hot_encoded = sources_one_hot_encoded

        # Set the selected gene columns to 0
        corrupted_data = self.data.clone()
        corrupted_data[:, missing_gene_indices] = 0
        self.corrupted_data = corrupted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.corrupted_data[idx], self.data[idx], self.sources_one_hot_encoded


def evaluate_model(model_BatchAE, model_src_adv, dataloader, device, parameters):
    model_BatchAE.eval()
    model_src_adv.eval()

    epoch_ae_loss_val = 0.0
    epoch_src_adv_loss_val = 0.0
    epoch_tot_loss_val = 0.0

    with torch.no_grad():
        for batch in dataloader:
            genes, labels = batch[0].to(device), batch[2].to(device)
            dec, enc, means, stdev = model_BatchAE(genes, labels)
            genes_not_corrupted = batch[1].to(device)

            v_loss = loss_function_vae(
                enc,
                dec,
                genes_not_corrupted,
                means,
                stdev,
                device=device,
                discrepancy_weight=parameters["discrepancy_weight"],
                discrepancy=parameters["discrepancy"],
            )

            src_pred = model_src_adv(enc)
            loss_src_adv = loss_function_classification(
                src_pred, labels, parameters["src_weights_src_adv"].to(device)
            )
            loss_ae = v_loss - parameters["src_adv_weight"] * loss_src_adv

            epoch_ae_loss_val += v_loss.detach().item()
            epoch_src_adv_loss_val += loss_src_adv.detach().item()
            epoch_tot_loss_val += loss_ae.detach().item()

    return epoch_ae_loss_val, epoch_src_adv_loss_val, epoch_tot_loss_val


class Alligner(BaseEstimator):
    def __init__(self, parameters):
        self.parameters = parameters
        self.device = torch.device(f"cuda:{self.parameters['selected_gpu']}")
        self.model_BatchAE = BatchVAE(
            n_genes=self.parameters["n_genes"],
            enc_dim=self.parameters["encoded_dim"],
            n_batch=self.parameters["n_sources"],
            enc_hidden_layers=(256, 128),
            enc_dropouts=(0.1, 0.1),
            dec_hidden_layers=(128, 256),
            dec_dropouts=(0.1, 0.1),
            batch_norm_enc=self.parameters["batch_norm_enc"],
            batch_norm_dec=self.parameters["batch_norm_dec"],
        )
        self.model_BatchAE.to(self.device)
        self.model_src_adv = MLP(
            input_dim=self.parameters["encoded_dim"],
            hidden_layers=self.parameters["mlp_hidden_layers"],
            output_dim=self.parameters["n_sources"],
            dropouts=self.parameters["mlp_dropouts"],
            batch_norm_mlp=self.parameters["batch_norm_mlp"],
            softmax_mlp=self.parameters["softmax_mlp"],
        )
        self.model_src_adv.to(self.device)

    def fit(self, ccle_features, tcga_features, must_keep_indices, verbose=False):
        """
        Fit the Mober model to the training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input training data.
        y : array-like, shape (n_samples,)
            The target values.

        Returns:
        self : object
            Returns self.


        """
        print("Create not corrupted datasets...")
        # Replace with your actual data sources
        ccle_dataset = CustomDataset(
            ccle_features, self.device, torch.tensor([1.0, 0.0])
        )
        tcga_dataset = CustomDataset(
            tcga_features, self.device, torch.tensor([0.0, 1.0])
        )
        print("Done")

        print("Create corrupted datasets...")
        indices_corrupted = get_indices_corrupted(
            self.parameters["n_genes"],
            must_keep_indices,
            missing_gene_percentage=self.parameters["missing_gene_percentage"],
        )
        ccle_corrupted_dataset = Corrupted_Dataset(
            ccle_features, self.device, torch.tensor([1.0, 0.0]), indices_corrupted
        )
        tcga_corrupted_dataset = Corrupted_Dataset(
            tcga_features, self.device, torch.tensor([0.0, 1.0]), indices_corrupted
        )
        print("Done")

        print("Concatenate datasets...")
        # Create a ConcatDataset to concatenate the two datasets
        combined_dataset = ConcatDataset([ccle_dataset, tcga_dataset])
        combined_dataset_with_corrupted = ConcatDataset(
            [ccle_dataset, tcga_dataset, ccle_corrupted_dataset, tcga_corrupted_dataset]
        )
        print("Done")

        print("Splits indices beteween train and val...")
        # Assuming you have a total of len(combined_dataset) samples
        total_samples = len(ccle_dataset) + len(tcga_dataset)

        # Extract labels from the dataset
        labels = [label for _, _, label, in combined_dataset]

        # Calculate the number of samples for each set
        num_train = int(self.parameters["train_ratio"] * total_samples)
        num_val = int(self.parameters["val_ratio"] * total_samples)

        if (
            self.parameters["train_ratio"]
            + self.parameters["val_ratio"]
            + self.parameters["test_ratio"]
            != 1
        ):
            raise ValueError(
                "The sum of train_ratio, val_ratio, and test_ratio must be equal to 1."
            )

        if self.parameters["test_ratio"] > 0:
            # Calculate the number of samples for each set
            num_test = total_samples - num_train - num_val

            # Use StratifiedSampler to split the dataset into training, validation, and test sets
            train_indices, rest_indices = train_test_split(
                range(total_samples), test_size=num_val + num_test, stratify=labels
            )
            val_indices, test_indices = train_test_split(
                rest_indices,
                test_size=num_test,
                stratify=[labels[i] for i in rest_indices],
            )

        else:
            test_dataloader = None
            # Use StratifiedSampler to split the dataset into training, validation, and test sets
            train_indices, val_indices = train_test_split(
                range(total_samples), test_size=num_val, stratify=labels
            )

        corrupted_train_indices = [i + total_samples for i in train_indices]
        corrupted_val_indices = [i + total_samples for i in val_indices]
        if self.parameters["test_ratio"] > 0:
            corrupted_test_indices = [i + total_samples for i in test_indices]

        train_indices_all = train_indices + corrupted_train_indices
        val_indices_all = val_indices + corrupted_val_indices
        if self.parameters["test_ratio"] > 0:
            test_indices_all = test_indices + corrupted_test_indices

        print("Done")

        print("Create samplers...")
        # Assuming you have train_indices, val_indices, test_indices
        train_sampler = StratifiedSampler(
            combined_dataset_with_corrupted, train_indices_all
        )
        val_sampler = StratifiedSampler(
            combined_dataset_with_corrupted, val_indices_all
        )
        if self.parameters["test_ratio"] > 0:
            test_sampler = StratifiedSampler(
                combined_dataset_with_corrupted, test_indices_all
            )
        print("Done")

        print("Create Datloaders...")
        # Create DataLoader instances for each set
        train_dataloader = DataLoader(
            combined_dataset_with_corrupted,
            batch_size=self.parameters["batch_size"],
            sampler=train_sampler,
        )
        val_dataloader = DataLoader(
            combined_dataset_with_corrupted,
            batch_size=self.parameters["batch_size"],
            sampler=val_sampler,
        )
        if self.parameters["test_ratio"] > 0:
            test_dataloader = DataLoader(
                combined_dataset_with_corrupted,
                batch_size=self.parameters["batch_size"],
                sampler=test_sampler,
            )
        print("Done")

        print("Create Optimizers...")
        # Define optimizers
        self.optimizer_BatchAE = getattr(
            optim, self.parameters["optimizer_name_BatchAE"]
        )(self.model_BatchAE.parameters(), lr=self.parameters["vae_learning_rate"])
        self.optimizer_src_adv = getattr(
            optim, self.parameters["optimizer_name_src_adv"]
        )(self.model_src_adv.parameters(), lr=self.parameters["adv_learning_rate"])
        print("Done")

        print("Starting training...")
        # Early stopping settings
        best_model_loss = float("inf")
        waited_epochs = 0
        early_stop = False

        for epoch in range(self.parameters["epochs"]):
            if early_stop:
                break

            epoch_ae_loss = 0.0
            epoch_src_adv_loss = 0.0
            epoch_tot_loss = 0.0

            self.model_BatchAE.train()
            self.model_src_adv.train()
            for batch in train_dataloader:
                genes, labels = batch[0].to(self.device), batch[2].to(self.device)
                genes_not_corrupted = batch[1].to(self.device)

                dec, enc, means, stdev = self.model_BatchAE(genes, labels)
                v_loss = loss_function_vae(
                    enc,
                    dec,
                    genes_not_corrupted,
                    means,
                    stdev,
                    device=self.device,
                    discrepancy_weight=self.parameters["discrepancy_weight"],
                    discrepancy=self.parameters["discrepancy"],
                )

                # Source adversary
                self.model_src_adv.zero_grad()

                src_pred = self.model_src_adv(enc)

                loss_src_adv = loss_function_classification(
                    src_pred,
                    labels,
                    self.parameters["src_weights_src_adv"].to(self.device),
                )

                loss_src_adv.backward(retain_graph=True)
                epoch_src_adv_loss += loss_src_adv.detach().item()
                self.optimizer_src_adv.step()

                src_pred = self.model_src_adv(enc)
                loss_src_adv = loss_function_classification(
                    src_pred,
                    labels,
                    self.parameters["src_weights_src_adv"].to(self.device),
                )

                # Update ae
                self.model_BatchAE.zero_grad()
                loss_ae = v_loss - self.parameters["src_adv_weight"] * loss_src_adv
                loss_ae.backward()
                epoch_ae_loss += v_loss.detach().item()
                self.optimizer_BatchAE.step()

                epoch_tot_loss += loss_ae.detach().item()

            avg_train_ae_loss = epoch_ae_loss / len(train_dataloader.dataset)
            avg_train_adv_loss = epoch_src_adv_loss / len(train_dataloader.dataset)
            avg_train_tot_loss = epoch_tot_loss / len(train_dataloader.dataset)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{self.parameters['epochs']} - Train AE Loss: {avg_train_ae_loss:.4f}, "
                    f"Train Adv Loss: {avg_train_adv_loss:.4f}, Train Total Loss: {avg_train_tot_loss:.4f}"
                )

            # Validation
            if val_dataloader is not None:
                epoch_ae_loss_val, epoch_src_adv_loss_val, epoch_tot_loss_val = (
                    evaluate_model(
                        self.model_BatchAE,
                        self.model_src_adv,
                        val_dataloader,
                        self.device,
                        self.parameters,
                    )
                )

                # Early stopping
                if epoch_ae_loss_val < best_model_loss:
                    best_model_loss = epoch_ae_loss_val
                    waited_epochs = 0
                else:
                    waited_epochs += 1
                    if waited_epochs >= self.parameters["early_stopping_patience"]:
                        early_stop = True
                        print("Early stopping triggered.")

                        # Validation
            if test_dataloader is not None:
                pass

        return self

    def alligne(self, features, space_to_project_into="TCGA"):
        """
        Alligne the data on the desired space (default TCGA).

        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        y_pred : array-like, shape (n_samples, n_features)
            The alligned data.
        """

        self.model_BatchAE.eval()
        genes = features.to(self.device)
        means, stdev, encoded_data = self.model_BatchAE.encoder(genes)
        # Create tensors with ones and zeros
        n_samples = features.shape[0]
        ones_column = torch.ones(n_samples, 1, device=self.device)
        zeros_column = torch.zeros(n_samples, 1, device=self.device)

        # Concatenate tensors along the second dimension (dim=1)
        if space_to_project_into == "CCLE":
            new_labels = torch.cat((ones_column, zeros_column), dim=1)
        elif space_to_project_into == "TCGA":
            new_labels = torch.cat((zeros_column, ones_column), dim=1)

        # Ensure that new_labels is moved to the same device as encoded_data
        new_labels = new_labels.to(self.device)

        decoded_data = self.model_BatchAE.decoder(encoded_data, new_labels)

        return decoded_data.cpu().detach().numpy()


class DataAccessor:
    def __init__(self):
        self.must_keep_indices = None
        self.ccle_final = None
        self.tcga_final = None

    def fit(
        self,
        ccle_data_path,
        tcga_projects_path,
        ccle_metadata_file_path,
        tcga_data_path,
        tcga_metadata_file_path,
        L1000_file_path,
    ):

        # Load data from files
        self.ccle_final, self.tcga_final, self.n_genes = access_data(
            ccle_data_path,
            tcga_projects_path,
            ccle_metadata_file_path,
            tcga_data_path,
            tcga_metadata_file_path,
        )

        # Open the file and read its contents line by line
        with open(L1000_file_path, "r") as file:
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

        df = self.tcga_final.iloc[:, -self.n_genes :]
        self.must_keep_indices = [
            idx for idx in df.columns.get_indexer(landmark_gene_symbols) if idx != -1
        ]

    def get_features(self, dataframe, device, standardize=True):
        indices = [
            idx
            for idx in dataframe.columns.get_indexer(
                self.ccle_final.iloc[:, -self.n_genes :].columns
            )
        ]
        # Create a DataFrame with all zeros
        if standardize:
            df = pd.DataFrame(np.zeros((len(dataframe), len(indices))))
        else:
            df = pd.DataFrame(np.full((len(dataframe), len(indices)), -1.0))
        # Fill the DataFrame with the values based on indices
        for i, index in enumerate(indices):
            if index != -1:
                df.iloc[:, i] = dataframe.iloc[:, index]

        features = load_data_lite(df, device, self.n_genes, standardize=standardize)
        return features


    def get_dataframe_from_features(self, features, original_dataframe):
        indices = [
            idx
            for idx in self.ccle_final.iloc[:,-self.n_genes:].columns.get_indexer(
                 original_dataframe.columns
            )
        ]
        
        # Create a DataFrame with all zeros
        df = original_dataframe.copy()

        # Fill the DataFrame with the values based on indices
        for i, index in enumerate(indices):
            if index != -1 and index < len(features):
                df.iloc[:, i] = features[:, index].detach().cpu().numpy()

        return df
