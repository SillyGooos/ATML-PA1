import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
# !pip install torchinfo
from torchinfo import summary
from tqdm.auto import tqdm
from pathlib import Path
# !pip install torch-fidelity
import torch_fidelity
# !pip install pytorch-fid
from pytorch_fid import fid_score
import torch.nn.functional as F
import os
import json

z_dim = 64
out_channels = 3
lr = 2e-4
epochs = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VAE(nn.Module):
    def __init__(self,in_channels=3, latent_dim=64, feature_map_enc=64, feature_map_dec=64):
        super(VAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, feature_map_enc, kernel_size=4, stride=2, padding=1)  # 64->32
        self.enc_bn1 = nn.BatchNorm2d(feature_map_enc)
        self.enc_conv2 = nn.Conv2d(feature_map_enc, feature_map_enc*2, kernel_size=4, stride=2, padding=1) # 32->16
        self.enc_bn2 = nn.BatchNorm2d(feature_map_enc*2)
        self.enc_conv3 = nn.Conv2d(feature_map_enc*2, feature_map_enc*4, kernel_size=4, stride=2, padding=1) # 16->8
        self.enc_bn3 = nn.BatchNorm2d(feature_map_enc*4)
        self.enc_conv4 = nn.Conv2d(feature_map_enc*4, feature_map_enc*8, kernel_size=4, stride=2, padding=1) # 8->4
        self.enc_bn4 = nn.BatchNorm2d(feature_map_enc*8)

        self.enc_fc = nn.Linear(feature_map_enc*8*4*4, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, feature_map_enc*8*4*4)

        #Decoder Conv Layers
        self.dec_deconv1 = nn.ConvTranspose2d(feature_map_dec*8, feature_map_dec*4, kernel_size=4, stride=2, padding=1) # 4->8
        self.dec_bn1 = nn.BatchNorm2d(feature_map_dec*4)
        self.dec_deconv2 = nn.ConvTranspose2d(feature_map_dec*4, feature_map_dec*2, kernel_size=4, stride=2, padding=1)  # 8->16
        self.dec_bn2 = nn.BatchNorm2d(feature_map_dec*2)
        self.dec_deconv3 = nn.ConvTranspose2d(feature_map_dec*2, feature_map_dec, kernel_size=4, stride=2, padding=1) # 16->32
        self.dec_bn3 = nn.BatchNorm2d(feature_map_dec)
        self.dec_deconv4 = nn.ConvTranspose2d(feature_map_dec, in_channels, kernel_size=4, stride=2, padding=1) # 32->64

        self._initialize_weights()


    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)

    def encode(self, x):
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), 0.2)
        x = F.leaky_relu(self.enc_bn2(self.enc_conv2(x)), 0.2)
        x = F.leaky_relu(self.enc_bn3(self.enc_conv3(x)), 0.2)
        x = F.leaky_relu(self.enc_bn4(self.enc_conv4(x)), 0.2)

        x = x.view(x.size(0), -1)
        hidden = F.relu(self.enc_fc(x))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        hidden = F.relu(self.dec_fc(z))
        hidden = F.relu(self.dec_fc2(hidden))
        hidden = hidden.view(-1, 512, 4, 4)
        hidden = F.relu(self.dec_deconv1(hidden))
        hidden = F.relu(self.dec_deconv2(hidden))
        hidden = F.relu(self.dec_deconv3(hidden))

        x_reconstructed = torch.tanh(self.dec_deconv4(hidden))
        return x_reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

