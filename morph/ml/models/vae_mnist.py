


import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_autoencoder import BaseAutoEncoder


class MNISTVAE(BaseAutoEncoder, nn.Module):
    def __init__(self, latent_dim=10):
#         print("MNIST VAE initialized")
        super().__init__()
        self.latent_dim = latent_dim

        # # ---------- Encoder ----------
        # self.fc1 = nn.Linear(28 * 28, 400)
        # self.fc_mu = nn.Linear(400, latent_dim)
        # self.fc_logvar = nn.Linear(400, latent_dim)

        # # ---------- Decoder ----------
        # self.fc3 = nn.Linear(latent_dim, 400)
        # self.fc4 = nn.Linear(400, 28 * 28)

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 28 * 28)


    # ---------- Encoder ----------
    def encode(self, x):
#         print("Encoding digit image")
#         return "latent_vector_digit"
        x = x.view(-1, 28 * 28)
        h = F.relu(self.fc1(x))
        # mu = self.fc_mu(h)
        # logvar = self.fc_logvar(h)
        # return mu, logvar
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    # ---------- Sampling ----------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------- Decoder ----------
    def decode(self, z):
        #print("Decoding digit latent vector")
        #return "reconstructed_digit_image"
        h = F.relu(self.fc3(z))
        # out = torch.sigmoid(self.fc4(h))
        # return out.view(-1, 1, 28, 28)
        h = F.relu(self.fc4(h))
        out = torch.sigmoid(self.fc5(h))
        return out.view(-1, 1, 28, 28)
    
    # ---------- Forward ----------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode_latent(self, x):
        """
        Returns only the latent mean (mu) for interpolation
        """
        mu, _ = self.encode(x)
        return mu

    def decode_latent(self, z):
        """
        Decode latent vector into image
        """
        return self.decode(z)
