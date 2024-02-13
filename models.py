import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list,
        output_dim: int,
        dropouts: list = None,
        batch_norm_mlp: bool = False,
        softmax_mlp: bool = False,
    ) -> None:
        """
        Initialize the Multi-Layer Perceptron (MLP) model.

        Args:
            input_dim (int): Dimensionality of input features.
            hidden_layers (list of int): List of hidden layer dimensions.
            output_dim (int): Dimensionality of output.
            dropouts (list of float, optional): List of dropout probabilities for each hidden layer. Defaults to None.
            batch_norm_mlp (bool, optional): Whether to use batch normalization. Defaults to False.
            softmax_mlp (bool, optional): Whether to apply softmax to output. Defaults to False.

        """
        super().__init__()
        self.batch_norm_mlp = batch_norm_mlp
        self.softmax_mlp = softmax_mlp
        self.activation = nn.SELU()
        layers = []
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if self.batch_norm_mlp:
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001))
            layers.append(nn.SELU())

            if dropouts and i < len(dropouts):
                layers.append(nn.Dropout(p=dropouts[i]))

            input_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.out_fc = nn.Linear(hidden_layers[-1], output_dim)
        if self.softmax_mlp:
            self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        out = self.layers(x)
        out = self.out_fc(out)
        if self.softmax_mlp:
            out = self.soft(out)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        enc_dim: int,
        hidden_layers: tuple = (256, 128),
        dropouts: tuple = (0.1, 0.1),
        batch_norm_enc: bool = False,
    ) -> None:
        """
        Initialize the Encoder model.

        Args:
            n_genes (int): Number of input genes.
            enc_dim (int): Dimensionality of the encoded output.
            hidden_layers (tuple of int, optional): Tuple of hidden layer dimensions. Defaults to (256, 128).
            dropouts (tuple of float, optional): Tuple of dropout probabilities. Defaults to (0.1, 0.1).
            batch_norm_enc (bool, optional): Whether to use batch normalization. Defaults to False.

        """
        super().__init__()

        self.batch_norm_enc = batch_norm_enc
        layers = []
        input_dim = n_genes
        for hidden_dim, dropout in zip(hidden_layers, dropouts):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if self.batch_norm_enc:
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001))
            layers.append(nn.SELU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.linear_means = nn.Linear(hidden_layers[-1], enc_dim)
        self.linear_log_vars = nn.Linear(hidden_layers[-1], enc_dim)

    def reparameterize(self, means: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize the samples from the latent distribution.

        Args:
            means (torch.Tensor): Mean values.
            stdev (torch.Tensor): Standard deviation values.

        Returns:
            torch.Tensor: Reparameterized samples.

        """
        return Normal(means, stdev).rsample()

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode the input data.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple of mean, standard deviation, and encoded output.

        """
        enc = self.layers(x)

        means = self.linear_means(enc)
        log_vars = self.linear_log_vars(enc)

        stdev = torch.exp(0.5 * log_vars) + 1e-4
        z = self.reparameterize(means, stdev)

        return means, stdev, z

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple of mean, standard deviation, and encoded output.

        """
        return self.encode(x)


class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        enc_dim: int,
        n_batch: int,
        hidden_layers: tuple = (128, 256),
        dropouts: tuple = (0.1, 0.1),
        batch_norm_dec: bool = False,
    ) -> None:
        """
        Initialize the Decoder model.

        Args:
            n_genes (int): Number of genes.
            enc_dim (int): Dimensionality of the encoded input.
            n_batch (int): Number of batches.
            hidden_layers (tuple of int, optional): Tuple of hidden layer dimensions. Defaults to (128, 256).
            dropouts (tuple of float, optional): Tuple of dropout probabilities. Defaults to (0.1, 0.1).
            batch_norm_dec (bool, optional): Whether to use batch normalization. Defaults to False.

        """
        super().__init__()

        self.batch_norm_dec = batch_norm_dec
        self.activation = nn.SELU()
        self.final_activation = nn.ReLU()

        # Additional batch processing
        self.fcb = nn.Linear(n_batch, n_batch)
        self.bnb = nn.BatchNorm1d(n_batch, momentum=0.01, eps=0.001)

        layers = []
        input_dim = enc_dim + n_batch
        for hidden_dim, dropout in zip(hidden_layers, dropouts):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if self.batch_norm_dec:
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001))
            layers.append(nn.SELU())
            input_dim = hidden_dim
            if dropout:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers)
        self.out_fc = nn.Linear(hidden_layers[-1], n_genes)

    def forward(self, z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder.

        Args:
            z (torch.Tensor): Encoded input tensor.
            batch (torch.Tensor): Batch information tensor.

        Returns:
            torch.Tensor: Decoded output tensor.

        """
        # Batch processing
        b = self.fcb(batch)
        b = self.bnb(b)
        b = self.activation(b)

        # Concatenate encoded input with batch information
        n_z = torch.cat([z, b], dim=1)

        # Decode layers
        dec = self.layers(n_z)
        dec = self.out_fc(dec)

        return dec


class BatchVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        enc_dim: int,
        n_batch: int,
        enc_hidden_layers: tuple,
        enc_dropouts: tuple,
        dec_hidden_layers: tuple,
        dec_dropouts: tuple,
        batch_norm_enc: bool,
        batch_norm_dec: bool,
    ) -> None:
        """
        Initialize the Batch Variational Autoencoder (VAE) model.

        Args:
            n_genes (int): Number of genes.
            enc_dim (int): Dimensionality of the encoded output.
            n_batch (int): Number of batches.
            enc_hidden_layers (tuple of int): Tuple of hidden layer dimensions for the encoder.
            enc_dropouts (tuple of float): Tuple of dropout probabilities for the encoder.
            dec_hidden_layers (tuple of int): Tuple of hidden layer dimensions for the decoder.
            dec_dropouts (tuple of float): Tuple of dropout probabilities for the decoder.
            batch_norm_enc (bool): Whether to use batch normalization for the encoder.
            batch_norm_dec (bool): Whether to use batch normalization for the decoder.

        """
        super().__init__()

        self.encoder = Encoder(
            n_genes, enc_dim, enc_hidden_layers, enc_dropouts, batch_norm_enc
        )
        self.decoder = Decoder(
            n_genes, enc_dim, n_batch, dec_hidden_layers, dec_dropouts, batch_norm_dec
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> tuple:
        """
        Forward pass of the Batch VAE.

        Args:
            x (torch.Tensor): Input tensor.
            batch (torch.Tensor): Batch information tensor.

        Returns:
            tuple: Tuple of decoded output, encoded input, means, and standard deviations.

        """
        means, stdev, enc = self.encoder(x)
        dec = self.decoder(enc, batch)

        return dec, enc, means, stdev


# Mober's paper version :
'''
class Encoder(nn.Module):
    """
    Encoder that takes the original gene expression and produces the encoding.

    Consists of 3 FC layers.
    """
    def __init__(self, n_genes, enc_dim):
        super().__init__()
        self.activation = nn.SELU()
        self.fc1 = nn.Linear(n_genes, 256)
        self.bn1 = nn.BatchNorm1d(256, momentum=0.01, eps=0.001)
        self.dp1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, momentum=0.01, eps=0.001)
        self.dp2 = nn.Dropout(p=0.1)
    
        self.linear_means = nn.Linear(128, enc_dim)
        self.linear_log_vars = nn.Linear(128, enc_dim)
    
    def reparameterize(self, means, stdev):
        
        return Normal(means, stdev).rsample()
        
    def encode(self, x):
        # encode
        enc = self.fc1(x)
        enc = self.bn1(enc)
        enc = self.activation(enc)
        enc = self.dp1(enc)
        enc = self.fc2(enc)
        enc = self.bn2(enc)
        enc = self.activation(enc)
        enc = self.dp2(enc)
        
        means = self.linear_means(enc)
        log_vars = self.linear_log_vars(enc)
        
        stdev = torch.exp(0.5 * log_vars) + 1e-4
        z = self.reparameterize(means, stdev)

        return means, stdev, z

    def forward(self, x):
        return self.encode(x)

    
class Decoder(nn.Module):
    """
    A decoder model that takes the encodings and a batch (source) matrix and produces decodings.

    Made up of 3 FC layers.
    """
    def __init__(self, n_genes, enc_dim, n_batch):
        super().__init__()
        self.activation = nn.SELU()
        self.final_activation = nn.ReLU()
        self.fcb = nn.Linear(n_batch, n_batch)
        self.bnb = nn.BatchNorm1d(n_batch, momentum=0.01, eps=0.001)
        self.fc4 = nn.Linear(enc_dim + n_batch, 128)
        self.bn4 = nn.BatchNorm1d(128, momentum=0.01, eps=0.001)
        self.fc5 = nn.Linear(128, 256)
        self.bn5 = nn.BatchNorm1d(256, momentum=0.01, eps=0.001)

        self.out_fc = nn.Linear(256, n_genes)


    def forward(self, z, batch):
        # batch input
        b = self.fcb(batch)
        b = self.bnb(b)
        b = self.activation(b)

        # concat with z
        n_z = torch.cat([z, b], dim=1)

        # decode layers
        dec = self.fc4(n_z)
        dec = self.bn4(dec)
        dec = self.activation(dec)
        dec = self.fc5(dec)
        dec = self.bn5(dec)
        dec = self.activation(dec)
        dec = self.final_activation(self.out_fc(dec))
        
        return dec


class BatchVAE(nn.Module):
    """
    Batch Autoencoder.
    Encoder is composed of 3 FC layers.
    Decoder is symmetrical to encoder + Batch input.
    """

    def __init__(self, n_genes, enc_dim, n_batch):
        super().__init__()

        self.encoder = Encoder(n_genes, enc_dim)
        self.decoder = Decoder(n_genes, enc_dim, n_batch)

    def forward(self, x, batch):
        means, stdev, enc = self.encoder(x)
        dec = self.decoder(enc, batch)
        
        return dec, enc, means, stdev
'''
