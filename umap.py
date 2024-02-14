import umap
import numpy as np
import pandas as pd
import plotly.express as px
from data_utils import index_to_abbreviation

def umap_genes(features_tcga, features_ccle, labels_tcga, labels_ccle):
    # Combined
    combined_features = np.concatenate([features_tcga, features_ccle], axis=0)
    # UMAP embedding for each feature set
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(combined_features)

    # Get the class labels (assuming `labels` is one-hot-encoded)
    class_labels_tcga = np.argmax(labels_tcga, axis=1)
    class_labels_ccle = np.argmax(labels_ccle, axis=1)
    
    labels_tcga_abb = [index_to_abbreviation[element.item()] for element in class_labels_tcga]
    labels_ccle_abb = [index_to_abbreviation[element.item()] for element in class_labels_ccle]


    # Ensure that class labels are unique
    class_labels_tcga_unique = pd.Series(labels_tcga_abb).drop_duplicates().tolist()
    class_labels_ccle_unique = pd.Series(labels_ccle_abb).drop_duplicates().tolist()

    
    # Create a unified color mapping for class labels
    class_color_mapping = {label: f'Class {label}' for label in set(class_labels_tcga_unique + class_labels_ccle_unique)}

    # Create DataFrame for Plotly
    df_tcga = pd.DataFrame({'UMAP Dimension 1': umap_emb[:len(labels_tcga_abb), 0], 'UMAP Dimension 2': umap_emb[:len(labels_tcga_abb), 1], 'Class Labels': labels_tcga_abb, 'Feature Set': 'TCGA'})
    df_ccle = pd.DataFrame({'UMAP Dimension 1': umap_emb[len(labels_tcga_abb):, 0], 'UMAP Dimension 2': umap_emb[len(labels_tcga_abb):, 1], 'Class Labels': labels_ccle_abb, 'Feature Set': 'CCLE'})
    df_combined = pd.concat([df_tcga, df_ccle])

    # Map class labels to colors using the unified color mapping
    df_combined['Color'] = df_combined['Class Labels'].map(class_color_mapping)

    # Plot with Plotly
    fig = px.scatter(df_combined, x='UMAP Dimension 1', y='UMAP Dimension 2', color='Color', symbol='Feature Set', size_max=10, labels={'Color': 'Class Labels'})
    fig.update_layout(title='UMAP Visualization of Gene Expression Features', xaxis_title='UMAP Dimension 1', yaxis_title='UMAP Dimension 2')

    # Show the interactive plot
    fig.show()