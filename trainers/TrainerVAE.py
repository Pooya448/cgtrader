import torch
import torch.optim as optim
from dataloader.ShapeNetDataset import ShapeNetDataset
import torch.nn.functional as F
from torch import nn
from models.VoxelVAE import VoxelVAE
from tqdm import tqdm


class TrainerVAE:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize dataset and dataloader
        self.dataset = ShapeNetDataset(args=config["data"])
        self.dataloader = self.dataset.get_loader()

        # Initialize model
        self.model = VoxelVAE(args=config["model"]).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=float(config["training"]["learning_rate"])
        )
        self.l2_lambda = float(config["training"].get("l2_lambda", 0.0))

    def weighted_binary_crossentropy(self, output, target):
        return (
            -(
                98.0 * target * torch.log(output)
                + 2.0 * (1.0 - target) * torch.log(1.0 - output)
            )
            / 100.0
        )

    def train_step_loss(self, recon_x, x, mu, logvar):

        recon_x = torch.clamp(torch.sigmoid(recon_x), 1e-7, 1.0 - 1e-7)

        # Voxel-wise Reconstruction Loss using weighted binary cross-entropy
        voxel_loss = torch.mean(self.weighted_binary_crossentropy(recon_x, x).float())

        # KL Divergence from isotropic Gaussian prior
        kl_div = -0.5 * torch.mean(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())

        return voxel_loss + kl_div
        # # BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction="mean")

        # # x = torch.clamp(x, min=-1, max=2)
        # # recon_x = torch.clamp(recon_x, min=0.1, max=1.0)
        # gamma = 0.97
        # # Modified BCE loss
        # BCE = -gamma * x * torch.log(recon_x) - (1 - gamma) * (1 - x) * torch.log(
        #     1 - recon_x
        # )
        # BCE = BCE.mean()

        # print(BCE)

        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # KLD = KLD.mean()

        # l2_reg = torch.tensor(0.0).to(self.device)
        # for param in self.model.parameters():
        #     l2_reg += torch.norm(param)

        # return BCE + KLD + self.l2_lambda * l2_reg

        # return BCE + KLD

    def train_step(self, data):
        data = data.to(self.device)
        self.optimizer.zero_grad()
        recon_batch, mu, logvar = self.model(data)
        loss = self.train_step_loss(recon_batch, data, mu, logvar)
        loss.backward()
        self.optimizer.step()
        return loss.item(), recon_batch, mu, logvar

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        p_bar = tqdm(
            self.dataloader,
            total=len(self.dataloader),
            desc=f"Epoch {epoch}/{self.config['training']['epochs']}",
        )
        for i, batch in enumerate(self.dataloader):
            data = batch["voxels"]
            loss, _, _, _ = self.train_step(data)
            train_loss += loss
            p_bar.set_postfix({"Loss": loss})
            p_bar.update(1)
            # print(f"Batch {i}/{len(self.dataloader)}, Loss: {loss:.6f}")
        return train_loss / len(self.dataloader.dataset)

    def run(self):
        for epoch in range(1, self.config["training"]["epochs"] + 1):
            train_loss = self.train(epoch)
            print(
                f'Epoch {epoch}/{self.config["training"]["epochs"]}, Loss: {train_loss:.4f}'
            )
        torch.save(self.model.state_dict(), "checkpoints/VAE/vae.pth")
