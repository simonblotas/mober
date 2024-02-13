import torch
from train import train_model
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import train_test_split
from data_utils import CustomDataset, StratifiedSampler, load_data
from torch import optim
from models import MLP, BatchVAE
from data_utils import access_data


parameters = {
    "vae_learning_rate": 5e-3,
    "adv_learning_rate": 5e-3,
    "epochs": 30,
    "n_genes": 18178,
    "encoded_dim": 64,
    "n_sources": 2,
    "src_weights_src_adv": torch.tensor([0.5, 1.0]),
    "kl_weight": 1e-5,
    "src_adv_weight": 0.01,
    "batch_size": 32,
    "early_stopping_patience": 5,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "Datasets": ["TCGA", "ACH"],
    "selected_gpu": 0,  # Choose the GPU index you want to use
    "space_to_project_into": "TCGA",
    "K": 10,  # Number of Nearest Neighboors considered
    "enc_hidden_layers": (256, 128),
    "enc_dropouts": (0.1, 0.1),
    "dec_hidden_layers": (128, 256),
    "dec_dropouts": (0.1, 0.1),
    "mlp_hidden_layers": (64, 64),
    "mlp_dropouts": None,
    "batch_norm_mlp": True,
    "softmax_mlp": True,
    "batch_norm_enc": True,
    "batch_norm_dec": True,
    "optimizer_name_src_adv": "Adam",
    "optimizer_name_BatchAE": "Adam",
}


ach_data_path = "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
tcga_data_path = "TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv"
tcga_projects_path = "TCGA_Projects.csv"
ach_metadata_file_path = "Model.csv"
tcga_metadata_file_path = "clinical.tsv"

# Create a torch.device object
device = torch.device(f"cuda:{parameters['selected_gpu']}")


ach_final, tcga_final = access_data(
    ach_data_path,
    tcga_projects_path,
    ach_metadata_file_path,
    tcga_data_path,
    tcga_metadata_file_path,
)


# Replace with your actual data sources
ach_dataset = CustomDataset(ach_final, device)
tcga_dataset = CustomDataset(tcga_final, device)

# Create a ConcatDataset to concatenate the two datasets
combined_dataset = ConcatDataset([ach_dataset, tcga_dataset])


# Assuming you have a total of len(combined_dataset) samples
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
optimizer_name_BatchAE = parameters["optimizer_name_BatchAE"]
optimizer_BatchAE = getattr(optim, optimizer_name_BatchAE)(
    model_BatchAE.parameters(), lr=parameters["vae_learning_rate"]
)
# model_src_adv = MLP(enc_dim =parameters['encoded_dim'] ,output_dim=parameters['n_sources'])
model_src_adv = MLP(
    input_dim=parameters["encoded_dim"],
    hidden_layers=parameters["mlp_hidden_layers"],
    output_dim=parameters["n_sources"],
    dropouts=parameters["mlp_dropouts"],
    batch_norm_mlp=parameters["batch_norm_mlp"],
    softmax_mlp=parameters["softmax_mlp"],
)
model_src_adv.to(device)
optimizer_name_src_adv = parameters["optimizer_name_src_adv"]
optimizer_src_adv = getattr(optim, optimizer_name_src_adv)(
    model_src_adv.parameters(), lr=parameters["adv_learning_rate"]
)

_, ach_abbreviations_labels, ach_features = load_data(ach_final, device)
_, tcga_abbreviations_labels, tcga_features = load_data(tcga_final, device)


train_model(
    model_BatchAE,
    optimizer_BatchAE,
    model_src_adv,
    optimizer_src_adv,
    device,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    ach_features,
    ach_abbreviations_labels,
    tcga_features,
    tcga_abbreviations_labels,
    parameters,
)
