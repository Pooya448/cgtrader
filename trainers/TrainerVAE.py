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
from visualize import save_voxel_as_mesh


class TrainerVAE:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize dataset and dataloader
        self.dataset = ShapeNetDataset(args=config["data"])
        self.train_loader = self.dataset.get_loader(train=True)
        self.test_loader = self.dataset.get_loader(train=False)

        # Initialize model
        self.model = VoxelVAE(args=config["model"]).to(self.device)

        self.learning_rate = float(config["training"]["learning_rate"])
        self.l2_lambda = config["training"].get("l2_lambda", 0.0)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.checkpoint_freq = config["training"]["checkpoint_freq"]
        self.visualize_freq = config["training"]["visualize_freq"]

        self.checkpoint_dir = Path("checkpoints")
        self.vis_dir = Path("vis")

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

        recon_x = torch.clamp(torch.sigmoid(recon_x), 1e-7, 1.0 - 1e-7)

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
                f"Epoch [{epoch}/{self.config['training']['epochs']}], Loss: {loss_epoch:.4f}, Voxel Loss: {voxel_loss_epoch:.4f}, KL Loss: {kl_loss_epoch:.4f}"
            )

        avg_loss = loss_epoch / len(self.train_loader)
        avg_voxel_loss = voxel_loss_epoch / len(self.train_loader)
        avg_kl_loss = kl_loss_epoch / len(self.train_loader)

        return avg_loss, avg_voxel_loss, avg_kl_loss

    def test(self):
        self.model.eval()

        test_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                data = batch["voxels"].to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss = self.step_loss(recon_batch, data, mu, logvar)
                test_loss += loss
        return test_loss / len(self.test_loader.dataset)

    def sample_and_generate_mesh(self, epoch):
        self.model.eval()

        with torch.no_grad():
            z = torch.randn(1, self.model.latent_dim).to(self.device)
            voxel = self.model.decode(z).cpu().numpy()
            voxel = voxel.squeeze()

            file_path = self.vis_dir / f"sampled_mesh_epoch-{epoch}.obj"
            save_voxel_as_mesh(voxel, file_path)

            wandb.log(
                {
                    "Sampled Mesh": [
                        wandb.Object3D(
                            file_path, caption=f"Sampled Mesh at Epoch {epoch}"
                        )
                    ]
                },
                step=epoch,
            )

    def sample_and_visualize(self, epoch):

        self.model.eval()

        with torch.no_grad():
            z = torch.randn(8, self.model.latent_dim).to(self.device)
            samples = self.model.decode(z).cpu().numpy()
            samples = samples.squeeze()

            fig = plt.figure(figsize=(15, 3))
            for i in range(samples.shape[0]):
                ax = fig.add_subplot(1, 8, i + 1, projection="3d")
                ax.voxels(samples[i] > 0.5, edgecolor="k")
                ax.set_axis_off()

            plt.suptitle(f"Samples at Epoch {epoch}")
            vis_path = f"vis/samples_epoch_{epoch}.png"
            plt.savefig(vis_path)
            plt.close(fig)

            wandb.log(
                {
                    "Samples": [
                        wandb.Image(vis_path, caption=f"Samples at Epoch {epoch}")
                    ]
                },
                step=epoch,
            )

    def run(self):

        for epoch in range(1, self.config["training"]["epochs"] + 1):
            avg_loss_epoch, avg_voxel_loss_epoch, avg_kl_loss_epoch = self.train_epoch(
                epoch
            )

            print(
                f"Epoch [{epoch}/{self.config['training']['epochs']}], Loss: {avg_loss_epoch:.4f}, Voxel Loss: {avg_voxel_loss_epoch:.4f}, KL Loss: {avg_kl_loss_epoch:.4f}"
            )

            wandb.log(
                {
                    "Epoch Train Loss": avg_loss_epoch,
                    "Epoch MSE Loss": avg_voxel_loss_epoch,
                    "Epoch KLD Loss": avg_kl_loss_epoch,
                },
                step=epoch,
            )

            if epoch % self.checkpoint_freq == 0:
                checkpoint_path = f"checkpoints/vae_epoch_{epoch}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")

            if epoch % self.visualize_freq == 0:
                self.sample_and_visualize(epoch)

        torch.save(self.model.state_dict(), "checkpoints/vae_final.pth")

        test_loss = self.test()
        print(f"Test Loss: {test_loss:.4f}")

        wandb.log_artifact(self.model)
