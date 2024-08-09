import torch
import torch.optim as optim
from dataloader.ShapeNetDataset import ShapeNetDataset
import torch.nn.functional as F
from torch import nn
from models.VoxelVAE import VoxelVAE
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import wandb
from visualize import save_voxel_as_mesh, crop_voxels
import numpy as np


class TrainerVAE:
    def __init__(self, config, run_name):
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Initialize dataset and dataloader
        self.dataset = ShapeNetDataset(args=config["data"])
        self.train_loader = self.dataset.get_loader(train=True)
        self.test_loader = self.dataset.get_loader(train=False)

        # Initialize model
        self.model = VoxelVAE(args=config["model"]).to(self.device)
        wandb.watch(self.model)

        self.learning_rate = float(config["training"]["learning_rate"])

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        self.checkpoint_freq = config["training"]["checkpoint_freq"]
        self.visualize_freq = config["training"]["visualize_freq"]

        self.checkpoint_dir = Path("checkpoints") / Path(run_name)
        self.vis_dir = Path("vis") / Path(run_name)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def step_loss(self, recon_x, x, mu, logvar):

        def weighted_binary_crossentropy(output, target):
            return (
                -(
                    98.0 * target * torch.log(output)
                    + 2.0 * (1.0 - target) * torch.log(1.0 - output)
                )
                / 100.0
            )

        recon_x = torch.clamp(recon_x, 1e-7, 1.0 - 1e-7)
        x = torch.clamp(recon_x, 1e-7, 1.0 - 1e-7)

        # Voxel-wise Reconstruction Loss using weighted binary cross-entropy
        voxel_loss = torch.mean(weighted_binary_crossentropy(recon_x, x).float())

        # KL Divergence from isotropic Gaussian prior
        kl_loss = -0.5 * torch.mean(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())

        return voxel_loss, kl_loss

    def train_step(self, data):
        data = data.to(self.device)
        self.optimizer.zero_grad()
        recon_batch, mu, logvar = self.model(data)
        voxel_loss_step, kl_loss_step = self.step_loss(recon_batch, data, mu, logvar)
        loss_step = voxel_loss_step + kl_loss_step
        loss_step.backward()
        self.optimizer.step()
        return (
            loss_step.item(),
            voxel_loss_step.item(),
            kl_loss_step.item(),
            recon_batch,
            mu,
            logvar,
        )

    def train_epoch(self, epoch):
        self.model.train()

        loss_epoch = 0
        voxel_loss_epoch = 0
        kl_loss_epoch = 0

        p_bar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
        )
        for i, batch in enumerate(p_bar):
            data = batch["voxels"]
            loss_step, voxel_loss_step, kl_loss_step, _, _, _ = self.train_step(data)

            loss_epoch += loss_step
            voxel_loss_epoch += voxel_loss_step
            kl_loss_epoch += kl_loss_step

            p_bar.set_description(
                f"Epoch [{epoch}/{self.config['training']['epochs']}], Step Loss: {loss_step:.4f}, Step Voxel Loss: {voxel_loss_step:.4f}, Step KL Loss: {kl_loss_step:.4f}"
            )

        avg_loss = loss_epoch / len(self.train_loader)
        avg_voxel_loss = voxel_loss_epoch / len(self.train_loader)
        avg_kl_loss = kl_loss_epoch / len(self.train_loader)

        return avg_loss, avg_voxel_loss, avg_kl_loss

    def test(self):
        self.model.eval()

        total_test_loss = 0
        total_kl_loss = 0
        total_voxel_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                data = batch["voxels"].to(self.device)
                recon_batch, mu, logvar = self.model(data)
                voxel_loss, kl_loss = self.step_loss(recon_batch, data, mu, logvar)

                total_kl_loss += kl_loss
                total_voxel_loss += voxel_loss
                total_test_loss += voxel_loss + kl_loss

        avg_test_loss = total_test_loss / len(self.test_loader.dataset)
        avg_voxel_loss = total_voxel_loss / len(self.test_loader.dataset)
        avg_kl_loss = total_kl_loss / len(self.test_loader.dataset)

        return avg_kl_loss, avg_voxel_loss, avg_test_loss

    def sample_and_generate_mesh(self, epoch):
        self.model.eval()

        with torch.no_grad():
            z = torch.randn(1, self.model.latent_dim).to(self.device)
            voxel = self.model.decode(z).cpu().numpy()
            voxel = voxel.squeeze()

            file_path = self.vis_dir / f"sampled_mesh_epoch-{epoch}.obj"
            save_voxel_as_mesh(voxel, file_path)

            wandb.log(
                {"Sampled Mesh": [wandb.Object3D(open(file_path))]},
                step=epoch,
            )

    def sample_and_visualize(self, epoch):

        self.model.eval()

        with torch.no_grad():
            z = torch.randn(4, self.model.latent_dim).to(self.device)
            samples = self.model.decode(z).cpu().numpy()
            samples = samples.squeeze()

            fig = plt.figure(figsize=(40, 20))

            for i in range(samples.shape[0]):
                sample = samples[i].copy()
                ax = fig.add_subplot(1, 4, i + 1, projection="3d")
                ax.voxels(sample > 0.5, facecolors="cyan", edgecolor="k")
                ax.set_axis_off()
                ax.view_init(elev=30, azim=45)

            plt.suptitle(f"Samples at Epoch {epoch}")
            vis_path = self.vis_dir / f"samples_epoch_{epoch}.png"
            plt.savefig(vis_path)
            plt.close(fig)

            wandb.log(
                {"Samples": [wandb.Image(str(vis_path))]},
                step=epoch,
            )

    def run(self):

        for epoch in range(1, self.config["training"]["epochs"] + 1):
            avg_loss_epoch, avg_voxel_loss_epoch, avg_kl_loss_epoch = self.train_epoch(
                epoch
            )

            print(
                f"Epoch [{epoch}/{self.config['training']['epochs']}], Epoch Loss: {avg_loss_epoch:.4f}, Epoch Voxel Loss: {avg_voxel_loss_epoch:.4f}, Epoch KL Loss: {avg_kl_loss_epoch:.4f}"
            )

            wandb.log(
                {
                    "Epoch Train Loss": avg_loss_epoch,
                    "Epoch BCE Loss": avg_voxel_loss_epoch,
                    "Epoch KLD Loss": avg_kl_loss_epoch,
                },
                step=epoch,
            )

            if epoch % self.checkpoint_freq == 0:
                checkpoint_path = self.checkpoint_dir / f"vae_epoch_{epoch}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")

            if epoch % self.visualize_freq == 0:
                self.sample_and_visualize(epoch)
                self.sample_and_generate_mesh(epoch)

        avg_kl_loss, avg_voxel_loss, avg_test_loss = self.test()
        print(
            f"Test Loss: {avg_test_loss:.4f}, Voxel (Reconstruction) Loss: {avg_voxel_loss:.4f}, KL Loss: {avg_kl_loss:.4f}"
        )
