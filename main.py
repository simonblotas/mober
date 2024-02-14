import torch
from train import train_model
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import train_test_split
from data_utils import CustomDataset, StratifiedSampler, load_data
from torch import optim
from models import MLP, BatchVAE
from data_utils import access_data

# Define hyperparameters for the training process
parameters = {
    "vae_learning_rate": 5e-3,  # Learning rate for the VAE model
    "adv_learning_rate": 5e-3,  # Learning rate for the source adversarial model
    "epochs": 30,  # Number of training epochs
    "n_genes": 18178,  # Number of genes in the dataset
    "encoded_dim": 64,  # Dimension of the latent space in the VAE model
    "n_sources": 2,  # Number of different sources in the dataset
    "src_weights_src_adv": torch.tensor(
        [0.5, 1.0]
    ),  # Weights for the source adversarial loss
    "kl_weight": 1e-5,  # Weight for the KL divergence loss in the VAE model
    "src_adv_weight": 0.01,  # Weight for the source adversarial loss
    "batch_size": 32,  # Batch size for training
    "early_stopping_patience": 5,  # Patience for early stopping during training
    "train_ratio": 0.7,  # Ratio of data used for training
    "val_ratio": 0.15,  # Ratio of data used for validation
    "test_ratio": 0.15,  # Ratio of data used for testing
    "Datasets": ["TCGA", "CCLE"],  # List of datasets used
    "selected_gpu": 0,  # Choose the GPU index you want to use
    "space_to_project_into": "TCGA",  # Space to project the data into ('TCGA' or 'CCLE')
    "K": 10,  # Number of nearest neighbors considered in metrics calculations
    "enc_hidden_layers": (256, 128),  # Sizes of hidden layers in the encoder
    "enc_dropouts": (0.1, 0.1),  # Dropout rates in the encoder
    "dec_hidden_layers": (128, 256),  # Sizes of hidden layers in the decoder
    "dec_dropouts": (0.1, 0.1),  # Dropout rates in the decoder
    "mlp_hidden_layers": (64, 64),  # Sizes of hidden layers in the MLP
    "mlp_dropouts": None,  # Dropout rates in the MLP (None for no dropout)
    "batch_norm_mlp": True,  # Whether to use batch normalization in the MLP
    "softmax_mlp": True,  # Whether to use softmax activation in the MLP
    "batch_norm_enc": True,  # Whether to use batch normalization in the encoder
    "batch_norm_dec": True,  # Whether to use batch normalization in the decoder
    "optimizer_name_src_adv": "Adam",  # Optimizer for the source adversarial model
    "optimizer_name_BatchAE": "Adam",  # Optimizer for the BatchAE model
}


# Paths to data files
ccle_data_path = "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
tcga_data_path = "TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv"
tcga_projects_path = "TCGA_Projects.csv"
ccle_metadata_file_path = "Model.csv"
tcga_metadata_file_path = "clinical.tsv"

# Create a torch.device object
device = torch.device(f"cuda:{parameters['selected_gpu']}")

# Load data from files
ccle_final, tcga_final = access_data(
    ccle_data_path,
    tcga_projects_path,
    ccle_metadata_file_path,
    tcga_data_path,
    tcga_metadata_file_path,
)


# Create Pytorch datasets
ccle_dataset = CustomDataset(ccle_final, device)
tcga_dataset = CustomDataset(tcga_final, device)

# Create a ConcatDataset to concatenate the two datasets
combined_dataset = ConcatDataset([ccle_dataset, tcga_dataset])

# Get the a total of len(combined_dataset) samples
total_samples = len(combined_dataset)

# Extract labels from the dataset
labels = [label for label, _, _ in combined_dataset]

# Calculate the number of samples for each set
num_train = int(parameters["train_ratio"] * total_samples)
num_val = int(parameters["val_ratio"] * total_samples)
num_test = total_samples - num_train - num_val

# Use StratifiedSampler to split the dataset into training, validation, and test sets
train_indices, rest_indices = train_test_split(
    range(total_samples), test_size=num_val + num_test, stratify=labels
)
val_indices, test_indices = train_test_split(
    rest_indices, test_size=num_test, stratify=[labels[i] for i in rest_indices]
)

"""
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

"""

# Assuming you have train_indices, val_indices, test_indices
train_sampler = StratifiedSampler(combined_dataset, train_indices)
val_sampler = StratifiedSampler(combined_dataset, val_indices)
test_sampler = StratifiedSampler(combined_dataset, test_indices)

# Create DataLoader instances for each set
train_dataloader = DataLoader(
    combined_dataset, batch_size=parameters["batch_size"], sampler=train_sampler
)
val_dataloader = DataLoader(
    combined_dataset, batch_size=parameters["batch_size"], sampler=val_sampler
)
test_dataloader = DataLoader(
    combined_dataset, batch_size=parameters["batch_size"], sampler=test_sampler
)

# Initialize the BatchVAE model
model_BatchAE = BatchVAE(
    n_genes=parameters["n_genes"],
    enc_dim=parameters["encoded_dim"],
    n_batch=parameters["n_sources"],
    enc_hidden_layers=(256, 128),
    enc_dropouts=(0.1, 0.1),
    dec_hidden_layers=(128, 256),
    dec_dropouts=(0.1, 0.1),
    batch_norm_enc=parameters["batch_norm_enc"],
    batch_norm_dec=parameters["batch_norm_dec"],
)
model_BatchAE.to(device)

# Initialize the optimizer for the BatchAE model
optimizer_name_BatchAE = parameters["optimizer_name_BatchAE"]
optimizer_BatchAE = getattr(optim, optimizer_name_BatchAE)(
    model_BatchAE.parameters(), lr=parameters["vae_learning_rate"]
)

# Initialize the MLP model for source adversarial training
model_src_adv = MLP(
    input_dim=parameters["encoded_dim"],
    hidden_layers=parameters["mlp_hidden_layers"],
    output_dim=parameters["n_sources"],
    dropouts=parameters["mlp_dropouts"],
    batch_norm_mlp=parameters["batch_norm_mlp"],
    softmax_mlp=parameters["softmax_mlp"],
)
model_src_adv.to(device)

# Initialize the optimizer for the source adversarial model
optimizer_name_src_adv = parameters["optimizer_name_src_adv"]
optimizer_src_adv = getattr(optim, optimizer_name_src_adv)(
    model_src_adv.parameters(), lr=parameters["adv_learning_rate"]
)

# Load ccle and TCGA data for metric compuation
_, ccle_abbreviations_labels, ccle_features = load_data(ccle_final, device)
_, tcga_abbreviations_labels, tcga_features = load_data(tcga_final, device)

# Train the models
train_model(
    model_BatchAE,
    optimizer_BatchAE,
    model_src_adv,
    optimizer_src_adv,
    device,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    ccle_features,
    ccle_abbreviations_labels,
    tcga_features,
    tcga_abbreviations_labels,
    parameters,
)