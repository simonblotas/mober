import numpy as np
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from data_utils import one_hot_to_abbreviation, index_to_abbreviation
import pandas as pd

def get_cancer_type(abbreviation, tcga_projects):
    # Convert abbreviation to uppercase for case-insensitive matching
    abbreviation = abbreviation.upper()
    
    # Find the row where the abbreviation matches
    row = tcga_projects[tcga_projects['Abbreviation'] == abbreviation]
    
    # If the row exists, return the corresponding cancer type, else return None
    if not row.empty:
        return row.iloc[0]['Cancer Type']
    else:
        return abbreviation


def fast_subtype_correlation_distances(cell_lines, tumors, cell_lines_subtypes, tumors_subtypes, tcga_project_file):
    correlation_coefficients = {}
    distances = {}

    tcga_projects = pd.read_csv(tcga_project_file)

    # Convert one-hot representation to abbreviation
    cell_lines_abbreviations = np.array([get_cancer_type(one_hot_to_abbreviation(torch.tensor(cell_lines_subtypes[element]), index_to_abbreviation), tcga_projects) for element in range(len(cell_lines_subtypes))])
    # Convert one-hot representation to abbreviation
    tumors_abbreviations = np.array([get_cancer_type(one_hot_to_abbreviation(torch.tensor(tumors_subtypes[element]), index_to_abbreviation), tcga_projects) for element in range(len(tumors_subtypes))])

    # Get unique subtypes
    unique_subtypes = np.unique(np.concatenate([cell_lines_abbreviations, tumors_abbreviations]))
    
    # Iterate over each subtype
    for subtype in unique_subtypes:
        # Select data for the current subtype
        cell_lines_subset = cell_lines[cell_lines_abbreviations == subtype]
        tumors_subset = tumors[tumors_abbreviations == subtype]


        # Compute distances and the Pearson correlation coefficients
        correlations = np.array([pearsonr(cell_line, tumor)[0] for cell_line, tumor in product(cell_lines_subset, tumors_subset)])
        distances_computed = np.array([np.linalg.norm(cell_line - tumor) for cell_line, tumor in product(cell_lines_subset, tumors_subset)])

        # Store the correlation coefficients for the current subtype
        correlation_coefficients[subtype] = correlations

        # Store the distance coefficients for the current subtype
        distances[subtype] = distances_computed

    return correlation_coefficients, distances



def plot_correlations(correlation_coefficients_mober,correlation_coefficients_celligner):
    # Iterate over each subtype and plot the correlation density for both methods
    for subtype, correlation_mober in correlation_coefficients_mober.items():
        correlation_celligner = correlation_coefficients_celligner[subtype]  # Get the corresponding correlation from the 'celligner' method
        
        # Flatten the correlation matrices into 1D arrays
        correlation_flat_mober = correlation_mober.flatten()
        correlation_flat_celligner = correlation_celligner.flatten()
        
        # Create a new figure for each subtype
        plt.figure(figsize=(8, 6))
        
        # Plot the correlation density using Seaborn's KDE plot for both methods
        sns.kdeplot(correlation_flat_mober, color='blue', linewidth=2.5, label='Mober')
        sns.kdeplot(correlation_flat_celligner, color='orange', linewidth=2.5, linestyle='--', label='Celligner')
        
        # Add title and labels
        plt.title(f'Correlation Density for Subtype {subtype}')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Density')
        
        # Show legend
        plt.legend()
        
        # Show the plot
        plt.show()

def plot_distances(distances_tcga,distances_celligner,distances_mober,distances_tcga_all):
    # Iterate over each subtype and plot the correlation density for both methods
    for subtype, d_tcga in distances_tcga.items():
        d_celligner = distances_celligner[subtype]  # Get the corresponding correlation from the 'celligner' method
        d_mober = distances_mober[subtype]  # Get the corresponding correlation from the 'celligner' method
        d_tcga_all = distances_tcga_all[subtype]
        
        


        # Flatten the correlation matrices into 1D arrays
        d_flat_tcga = d_tcga.flatten()
        d_flat_celligner = d_celligner.flatten()
        d_flat_mober = d_mober.flatten()
        d_flat_tcga_all = d_tcga_all.flatten()
        
        # Create a new figure for each subtype
        plt.figure(figsize=(8, 6))
        
        # Plot the correlation density using Seaborn's KDE plot for both methods
        sns.kdeplot(d_flat_tcga, color='blue', linewidth=2.5, label='Distances TCGA <-> TCGA Mober (samplewise)')
        sns.kdeplot(d_flat_celligner, color='orange', linewidth=2.5, linestyle='--', label='Distances CCLE <-> TCGA Celligner (pairwise)')
        sns.kdeplot(d_flat_mober, color='red', linewidth=2.5, linestyle='--', label='Distances CCLE <-> TCGA Mober (pairwise)')
        sns.kdeplot(d_flat_tcga_all, color='green', linewidth=2.5, linestyle='--', label='Distances TCGA <-> TCGA Mober (pairwise)')
        
        # Add title and labels
        plt.title(f'Distances Density for Subtype {subtype}')
        plt.xlabel('L2 distances')
        plt.ylabel('Density')
        
        # Show legend
        plt.legend()
        
        # Show the plot
        plt.show()
