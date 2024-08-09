import argparse
import torch
from models.VoxelVAE import VoxelVAE
from pathlib import Path
import matplotlib.pyplot as plt
from visualize import save_voxel_as_mesh
import yaml


### Takes a model checkpoint path and generates samples from the trained model
class InferVAE:
    def __init__(
        self,
        config_path="config/vae.yaml",
        checkpoint_path=None,
        output_dir="infer_output",
    ):

        if checkpoint_path is None:
            raise ValueError("You must provide a valid checkpoint_path.")

        ### Load the config
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VoxelVAE(args=self.config["model"]).to(self.device)

        ### Load the weights to the model
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    ### Samples from normal distribution and feeds it through the decoder
    def sample(self, num_samples=8):
        with torch.no_grad():
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            samples = self.model.decode(z).cpu().numpy()
        return samples

    ### Generates meshes and plots by sampling from the latent space
    def generate_and_save(self, num_samples=8):
        samples = self.sample(num_samples=num_samples)
        samples = samples.squeeze()

        for i in range(samples.shape[0]):
            voxel_grid = samples[i]

            file_path = self.output_dir / f"sample_{i}.obj"
            save_voxel_as_mesh(voxel_grid, file_path)
            print(f"Saved sample {i} as {file_path}")

            self.save_voxel_image(voxel_grid, i)

    def save_voxel_image(self, voxel_grid, index):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.voxels(voxel_grid > 0.5, facecolors="cyan", edgecolor="k")
        ax.set_axis_off()
        ax.view_init(elev=30, azim=45)
        plt.savefig(self.output_dir / f"sample_{index}.png")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer VAE and generate 3D objects.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/vae.yaml",
        help="Path to the config file (default: config/vae.yaml)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint file (e.g., checkpoints/vae_final.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="infer_output",
        help="Directory where generated samples will be saved (default: infer_output)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples to generate (default: 8)",
    )

    args = parser.parse_args()

    infer_vae = InferVAE(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
    )
    infer_vae.generate_and_save(num_samples=args.num_samples)
