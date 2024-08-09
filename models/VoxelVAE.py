import torch
import torch.nn as nn
import torch.nn.functional as F


### A simple building block of the encoder network.
### Convolution -> BatchNormalization -> Activation
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


### A simple building block of the encoder network.
### TransposeConvolution -> BatchNormalization -> Activation
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        output_padding=0,
        is_last_layer=False,
    ):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding=output_padding,
        )
        self.is_last_layer = is_last_layer

        ### If this is the last layer in the decoder network, we don't want any ELUactivations
        if not self.is_last_layer:
            self.norm = nn.BatchNorm3d(out_channels)
            self.act = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        if not self.is_last_layer:
            x = self.norm(x)
            x = self.act(x)
        return x


### Linear module for latent code
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, use_batchnorm=True):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


class VoxelVAE(nn.Module):
    def __init__(self, args):
        super(VoxelVAE, self).__init__()
        self.latent_dim = args["latent_dim"]

        # Encoder
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    1, 8, kernel_size=3, stride=1, padding=0
                ),  # 32x32x32 -> 30x30x30
                EncoderBlock(
                    8, 16, kernel_size=3, stride=2, padding=1
                ),  # 30x30x30 -> 15x15x15
                EncoderBlock(
                    16, 32, kernel_size=3, stride=1, padding=0
                ),  # 15x15x15 -> 13x13x13
                EncoderBlock(
                    32, 64, kernel_size=3, stride=2, padding=1
                ),  # 13x13x13 -> 7x7x7
            ]
        )

        self.fc_mu = LinearBlock(64 * 7 * 7 * 7, self.latent_dim)
        self.fc_logvar = LinearBlock(64 * 7 * 7 * 7, self.latent_dim)

        # Decoder
        self.fc_decode = LinearBlock(self.latent_dim, 64 * 7 * 7 * 7)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    64, 64, kernel_size=3, stride=1, padding=1
                ),  # 7x7x7 -> 7x7x7
                DecoderBlock(
                    64, 32, kernel_size=3, stride=2, padding=0
                ),  # 7x7x7 -> 15x15x15
                DecoderBlock(
                    32, 16, kernel_size=3, stride=1, padding=1
                ),  # 15x15x15 -> 15x15x15
                DecoderBlock(
                    16, 8, kernel_size=4, stride=2, padding=0, output_padding=0
                ),  # 15x15x15 -> 32x32x32
                DecoderBlock(
                    8, 1, kernel_size=3, stride=1, padding=1, is_last_layer=True
                ),  # 32x32x32 -> 32x32x32 (output layer)
            ]
        )

    def encode(self, x):
        for block in self.encoder:
            x = block(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    ### Does the reparametrization technic in training VAEs, ensuring a differentiability through the stochastic sampling process
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 64, 7, 7, 7)
        for block in self.decoder:
            x = block(x)
        x = torch.sigmoid(x)  ### Ensure output is in [0, 1]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
