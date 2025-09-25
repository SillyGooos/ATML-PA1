import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn.functional as F
import os
from pathlib import Path
import json
# !pip install pytorch-fid
# !pip install torchinfo
from pytorch_fid import fid_score
from torchinfo import summary

z_dim = 64
feature_map = 64
output_channels = 3
lr = 2e-4
epochs = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Generator(torch.nn.Module):
    def __init__(self, z_dim = z_dim, feature_map = feature_map, output_channels = 3):
        super().__init__()
        self.gan_gen = torch.nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_map*8, kernel_size=4, stride=1, padding=0, bias=False), #Start with a rich network (i.e featuremap * 8)
            nn.BatchNorm2d(feature_map*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map*8, feature_map*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map*4, feature_map*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map*2, feature_map, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gan_gen(z)

class Discriminator(nn.Module):
    def __init__(self, image_channels =3, feature_map = 64):
        super().__init__()
        self.gan_disc = nn.Sequential(
            nn.Conv2d(image_channels, feature_map, kernel_size=4,stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #Shape = (batchsize, 64,16,16)

            nn.Conv2d(feature_map, feature_map*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map*2),
            nn.LeakyReLU(0.2, inplace=True),
            #Shape = (batchsize, 128,8,8)

            nn.Conv2d(feature_map*2, feature_map*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map*4),
            nn.LeakyReLU(0.2, inplace=True),
            #Shape = (batchsize, 256,4,4)

            nn.Conv2d(feature_map*4, feature_map*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map*8),
            nn.LeakyReLU(0.2, inplace=True),
            #Shape = (batchsize, 512,2,2)

            nn.Conv2d(feature_map*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            #output(batchsize, 1,1,1)-> probability of "real"
        )
    def forward(self, x):
        return self.gan_disc(x).flatten(start_dim=1)
