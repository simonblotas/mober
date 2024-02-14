import torch
from loss import loss_function_vae, loss_function_classification
import wandb
from datetime import datetime
from metrics import project_into_decoded_space, metrics_on_dataloader
import numpy as np
from typing import Tuple, Optional


def evaluate_model(
    model_BatchAE: torch.nn.Module,
    model_src_adv: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    parameters: dict,
) -> Tuple[float, float, float]:
    """
    Evaluate the model.

    Parameters:
    - model_BatchAE (torch.nn.Module): BatchAE model.
    - model_src_adv (torch.nn.Module): Source adversarial model.
    - dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
    - device (torch.device): Device to perform evaluation on.
    - parameters (dict): Dictionary containing evaluation parameters.

    Returns:
    - Tuple[float, float, float]: Tuple containing epoch AE loss, epoch source adversarial loss, and total epoch loss.
    """

    # Set models to evaluation mode
    model_BatchAE.eval()
    model_src_adv.eval()

    # Initialize variables to store loss values
    epoch_ae_loss_val = 0.0
    epoch_src_adv_loss_val = 0.0
    epoch_tot_loss_val = 0.0

    # Disable gradient computation during evaluation
    with torch.no_grad():
        # Iterate over batches in the dataloader
        for batch in dataloader:
            genes, labels = batch[2].to(device), batch[0].to(device)
            # Forward pass through BatchAE model
            dec, enc, means, stdev = model_BatchAE(genes, labels)
            # Compute VAE loss
            v_loss = loss_function_vae(
                dec, genes, means, stdev, kl_weight=parameters["kl_weight"]
            )

            # Forward pass through source adversarial model
            src_pred = model_src_adv(enc)
            # Compute source adversarial loss
            loss_src_adv = loss_function_classification(
                src_pred, labels, parameters["src_weights_src_adv"].to(device)
            )
            # Compute total loss
            loss_ae = v_loss - parameters["src_adv_weight"] * loss_src_adv

            # Accumulate loss values
            epoch_ae_loss_val += v_loss.detach().item()
            epoch_src_adv_loss_val += loss_src_adv.detach().item()
            epoch_tot_loss_val += loss_ae.detach().item()

    return epoch_ae_loss_val, epoch_src_adv_loss_val, epoch_tot_loss_val


def metrics_on_model(
    model_BatchAE: torch.nn.Module,
    tcga_abbreviations_labels: torch.Tensor,
    tcga_features_contiguous: torch.Tensor,
    ccle_abbreviations_labels: torch.Tensor,
    ccle_features_contiguous: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    parameters: dict,
) -> Tuple[float, float, float]:
    """
    Compute metrics on the model.

    Parameters:
    - model_BatchAE (torch.nn.Module): BatchAE model.
    - tcga_abbreviations_labels (torch.Tensor): Abbreviations labels for TCGA data.
    - tcga_features_contiguous (torch.Tensor): Contiguous features for TCGA data.
    - ccle_abbreviations_labels (torch.Tensor): Abbreviations labels for CCLE data.
    - ccle_features_contiguous (torch.Tensor): Contiguous features for CCLE data.
    - dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
    - device (torch.device): Device to perform evaluation on.
    - parameters (dict): Dictionary containing evaluation parameters.

    Returns:
    - Tuple[float, float, float]: Accuracy on all data, accuracy on CCLE data, accuracy on TCGA data.
    """

    # Metric on set :
    (
        all_decoded_data,
        true_abbreviations,
        ccle_decoded_data,
        ccle_true_abbreviations,
        tcga_decoded_data,
        tcga_true_abbreviations,
    ) = project_into_decoded_space(
        dataloader, model_BatchAE, device, parameters["space_to_project_into"]
    )

    if parameters["space_to_project_into"] == "TCGA":
        # Calculate accuracies for TCGA data
        acc_all_data = metrics_on_dataloader(
            tcga_abbreviations_labels.numpy(),
            tcga_features_contiguous,
            true_abbreviations,
            all_decoded_data,
        )
        acc_ccle = metrics_on_dataloader(
            tcga_abbreviations_labels.numpy(),
            tcga_features_contiguous,
            ccle_true_abbreviations,
            ccle_decoded_data,
        )
        acc_tcga = metrics_on_dataloader(
            tcga_abbreviations_labels.numpy(),
            tcga_features_contiguous,
            tcga_true_abbreviations,
            tcga_decoded_data,
        )
    elif parameters["space_to_project_into"] == "CCLE":
        # Calculate accuracies for CCLE data
        acc_all_data = metrics_on_dataloader(
            ccle_abbreviations_labels.numpy(),
            ccle_features_contiguous,
            true_abbreviations,
            all_decoded_data,
        )
        acc_ccle = metrics_on_dataloader(
            tcga_abbreviations_labels.numpy(),
            tcga_features_contiguous,
            ccle_true_abbreviations,
            ccle_decoded_data,
        )
        acc_tcga = metrics_on_dataloader(
            tcga_abbreviations_labels.numpy(),
            tcga_features_contiguous,
            tcga_true_abbreviations,
            tcga_decoded_data,
        )

    return acc_all_data, acc_ccle, acc_tcga


def train_model(
    model_BatchAE: torch.nn.Module,
    optimizer_BatchAE: torch.optim.Optimizer,
    model_src_adv: torch.nn.Module,
    optimizer_src_adv: torch.optim.Optimizer,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader],
    test_dataloader: Optional[torch.utils.data.DataLoader],
    ccle_features: np.ndarray,
    ccle_abbreviations_labels: torch.Tensor,
    tcga_features: np.ndarray,
    tcga_abbreviations_labels: torch.Tensor,
    parameters: dict,
) -> float:
    """
    Train the model and evaluate its performance.

    Parameters:
    - model_BatchAE (torch.nn.Module): BatchAE model.
    - optimizer_BatchAE (torch.optim.Optimizer): Optimizer for BatchAE.
    - model_src_adv (torch.nn.Module): Source adversary model.
    - optimizer_src_adv (torch.optim.Optimizer): Optimizer for source adversary.
    - device (torch.device): Device to perform training and evaluation on.
    - train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
    - val_dataloader (Optional[torch.utils.data.DataLoader]): DataLoader for validation data.
    - test_dataloader (Optional[torch.utils.data.DataLoader]): DataLoader for test data.
    - ccle_features (np.ndarray): Features for CCLE data.
    - ccle_abbreviations_labels (torch.Tensor): Abbreviations labels for CCLE data.
    - tcga_features (np.ndarray): Features for TCGA data.
    - tcga_abbreviations_labels (torch.Tensor): Abbreviations labels for TCGA data.
    - parameters (dict): Dictionary containing training parameters.

    Returns:
    - float: Accuracy on CCLE data.
    """

    tcga_features_contiguous = np.ascontiguousarray(tcga_features)
    ccle_features_contiguous = np.ascontiguousarray(ccle_features)

    # Initialize WandB run

    # Get the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a dynamic project name using the current date and time
    traning_name = f"training_run_{current_time}"
    wandb.init(
        project="Mober",
        name=traning_name,
        config={
            "vae_learning_rate": parameters["vae_learning_rate"],
            "adv_learning_rate": parameters["adv_learning_rate"],
            "n_genes": parameters["n_genes"],
            "batch_size": parameters["batch_size"],
            "encoded_dim": parameters["encoded_dim"],
            "n_sources": parameters["n_sources"],
            "epochs": parameters["epochs"],
            "kl_weight": parameters["kl_weight"],
            "early_stopping_patience": parameters["early_stopping_patience"],
            "src_adv_weight": parameters["src_adv_weight"],
            "src_weights_src_adv": parameters["src_weights_src_adv"].tolist(),
            "train_ratio": parameters["train_ratio"],
            "val_ratio": parameters["val_ratio"],
            "test_ratio": parameters["test_ratio"],
            "Datasets": parameters["Datasets"],
            "space_to_project_into": parameters["space_to_project_into"],
            "K": parameters["K"],  # Number of Nearest Neighboors considered
            "enc_hidden_layers": parameters["enc_hidden_layers"],
            "enc_dropouts": parameters["enc_dropouts"],
            "dec_hidden_layers": parameters["dec_hidden_layers"],
            "dec_dropouts": parameters["dec_dropouts"],
            "mlp_hidden_layers": parameters["mlp_hidden_layers"],
            "mlp_dropouts": parameters["mlp_dropouts"],
            "optimizer_name_src_adv": parameters["optimizer_name_src_adv"],
            "optimizer_name_BatchAE": parameters["optimizer_name_src_adv"],
            "batch_norm_mlp": parameters["batch_norm_mlp"],
            "softmax_mlp": parameters["softmax_mlp"],
            "batch_norm_enc": parameters["batch_norm_enc"],
            "batch_norm_dec": parameters["batch_norm_dec"],
        },
    )

    # Early stopping settings
    best_model_loss = float("inf")
    waited_epochs = 0
    early_stop = False

    for epoch in range(parameters["epochs"]):
        if early_stop:
            break

        epoch_ae_loss = 0.0
        epoch_src_adv_loss = 0.0
        epoch_tot_loss = 0.0

        model_BatchAE.train()
        model_src_adv.train()
        for batch in train_dataloader:
            genes, labels = batch[2].to(device), batch[0].to(device)

            dec, enc, means, stdev = model_BatchAE(genes, labels)
            v_loss = loss_function_vae(
                dec, genes, means, stdev, kl_weight=parameters["kl_weight"]
            )

            # Source adversary
            model_src_adv.zero_grad()

            src_pred = model_src_adv(enc)

            loss_src_adv = loss_function_classification(
                src_pred, labels, parameters["src_weights_src_adv"].to(device)
            )
            loss_src_adv.backward(retain_graph=True)
            epoch_src_adv_loss += loss_src_adv.detach().item()
            optimizer_src_adv.step()

            src_pred = model_src_adv(enc)
            loss_src_adv = loss_function_classification(
                src_pred, labels, parameters["src_weights_src_adv"].to(device)
            )

            # Update ae
            model_BatchAE.zero_grad()
            loss_ae = v_loss - parameters["src_adv_weight"] * loss_src_adv
            loss_ae.backward()
            epoch_ae_loss += v_loss.detach().item()
            optimizer_BatchAE.step()

            epoch_tot_loss += loss_ae.detach().item()

        avg_train_ae_loss = epoch_ae_loss / len(train_dataloader.dataset)
        avg_train_adv_loss = epoch_src_adv_loss / len(train_dataloader.dataset)
        avg_train_tot_loss = epoch_tot_loss / len(train_dataloader.dataset)

        # Log metrics to WandB

        wandb.log(
            {
                "Train/AE_Loss": avg_train_ae_loss,
                "Train/Adv_Loss": avg_train_adv_loss,
                "Train/Total_Loss": avg_train_tot_loss,
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch + 1}/{parameters['epochs']} - Train AE Loss: {avg_train_ae_loss:.4f}, "
            f"Train Adv Loss: {avg_train_adv_loss:.4f}, Train Total Loss: {avg_train_tot_loss:.4f}"
        )

        # Metric on training set :
        train_acc_all_data, train_acc_ccle, train_acc_tcga = metrics_on_model(
            model_BatchAE,
            tcga_abbreviations_labels,
            tcga_features_contiguous,
            ccle_abbreviations_labels,
            ccle_features_contiguous,
            train_dataloader,
            device,
            parameters,
        )
        wandb.log(
            {
                "Train_Accuracy_all_data_on_"
                + parameters["space_to_project_into"]: train_acc_all_data
            },
            step=epoch,
        )
        wandb.log(
            {
                "Train_Accuracy_ach_on_"
                + parameters["space_to_project_into"]: train_acc_ccle
            },
            step=epoch,
        )
        wandb.log(
            {
                "Train_Accuracy_tcga_on_"
                + parameters["space_to_project_into"]: train_acc_tcga
            },
            step=epoch,
        )

        # Validation
        if val_dataloader is not None:
            epoch_ae_loss_val, epoch_src_adv_loss_val, epoch_tot_loss_val = (
                evaluate_model(
                    model_BatchAE, model_src_adv, val_dataloader, device, parameters
                )
            )
            wandb.log(
                {
                    "Validation/Total_Loss": epoch_tot_loss_val
                    / len(val_dataloader.dataset)
                },
                step=epoch,
            )
            wandb.log(
                {"Validation/Loss_ae": epoch_ae_loss_val / len(val_dataloader.dataset)},
                step=epoch,
            )
            wandb.log(
                {
                    "Validation/Loss_adv": epoch_src_adv_loss_val
                    / len(val_dataloader.dataset)
                },
                step=epoch,
            )

            # Early stopping
            if epoch_ae_loss_val < best_model_loss:
                best_model_loss = epoch_ae_loss_val
                waited_epochs = 0
            else:
                waited_epochs += 1
                if waited_epochs >= parameters["early_stopping_patience"]:
                    early_stop = True
                    print("Early stopping triggered.")

            val_acc_all_data, val_acc_ccle, val_acc_tcga = metrics_on_model(
                model_BatchAE,
                tcga_abbreviations_labels,
                tcga_features_contiguous,
                ccle_abbreviations_labels,
                ccle_features_contiguous,
                val_dataloader,
                device,
                parameters,
            )
            wandb.log(
                {
                    "Validation_Accuracy_all_data_on_"
                    + parameters["space_to_project_into"]: val_acc_all_data
                },
                step=epoch,
            )
            wandb.log(
                {
                    "Validation_Accuracy_ach_on_"
                    + parameters["space_to_project_into"]: val_acc_ccle
                },
                step=epoch,
            )
            wandb.log(
                {
                    "Validation_Accuracy_tcga_on_"
                    + parameters["space_to_project_into"]: val_acc_tcga
                },
                step=epoch,
            )

        if test_dataloader is not None:
            epoch_ae_loss_test, epoch_src_adv_loss_test, epoch_tot_loss_test = (
                evaluate_model(
                    model_BatchAE, model_src_adv, test_dataloader, device, parameters
                )
            )
            wandb.log(
                {"Test/Total_Loss": epoch_tot_loss_test / len(test_dataloader.dataset)},
                step=epoch,
            )
            wandb.log(
                {"Test/Loss_ae": epoch_ae_loss_test / len(test_dataloader.dataset)},
                step=epoch,
            )
            wandb.log(
                {
                    "Test/Loss_adv": epoch_src_adv_loss_test
                    / len(test_dataloader.dataset)
                },
                step=epoch,
            )

            test_acc_all_data, test_acc_ccle, test_acc_tcga = metrics_on_model(
                model_BatchAE,
                tcga_abbreviations_labels,
                tcga_features_contiguous,
                ccle_abbreviations_labels,
                ccle_features_contiguous,
                test_dataloader,
                device,
                parameters,
            )
            wandb.log(
                {
                    "Test_Accuracy_all_data_on_"
                    + parameters["space_to_project_into"]: test_acc_all_data
                },
                step=epoch,
            )
            wandb.log(
                {
                    "Test_Accuracy_ach_on_"
                    + parameters["space_to_project_into"]: test_acc_ccle
                },
                step=epoch,
            )
            wandb.log(
                {
                    "Test_Accuracy_tcga_on_"
                    + parameters["space_to_project_into"]: test_acc_tcga
                },
                step=epoch,
            )

    wandb.finish()  # Finish WandB run
    del model_BatchAE
    del model_src_adv

    return test_acc_ccle
