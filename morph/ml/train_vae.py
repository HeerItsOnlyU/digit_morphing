import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.vae_mnist import MNISTVAE

# ---------- Hyperparameters ----------
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3
LATENT_DIM = 10
KL_WEIGHT = 0.0




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Loss Function ----------
# def vae_loss(recon_x, x, mu, logvar):
#     # Reconstruction loss
#     BCE = F.binary_cross_entropy(
#         recon_x.view(-1, 784),
#         x.view(-1, 784),
#         reduction='sum'
#     )

#     # KL divergence
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD
def vae_loss(recon_x, x, mu, logvar, kl_weight):
    BCE = F.binary_cross_entropy(
        recon_x.view(-1, 784),
        x.view(-1, 784),
        reduction='sum'
    )

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kl_weight * KLD



# ---------- Training Function ----------
def train():
    # MNIST Dataset
    transform = transforms.ToTensor()
    # transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
    # ])


    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Model
    model = MNISTVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    best_loss = float("inf")
    for epoch in range(EPOCHS):
        total_loss = 0
        kl_weight = min(1.0, epoch / 10)

        for images, _ in train_loader:
            images = images.to(DEVICE)

            optimizer.zero_grad()

            recon_images, mu, logvar = model(images)
            loss = vae_loss(recon_images, images, mu, logvar,kl_weight)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.2f}")

    # Save model
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), "mnist_vae.pth")
        print("✅ Best model updated and saved")



if __name__ == "__main__":
    train()
