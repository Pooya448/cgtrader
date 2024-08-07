import argparse
from config.parse import load_config
from trainers.TrainerVAE import TrainerVAE
import wandb


def main(model_type):
    config_path = f"config/{model_type.lower()}.yaml"
    config = load_config(config_path)

    if model_type.lower() == "vae":
        trainer = TrainerVAE(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Initialize wandb
    wandb.init(project="3DGen-Task", config=config)
    config = wandb.config

    trainer.run()

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "model_type", type=str, help="Type of model to train (e.g., VAE)"
    )
    args = parser.parse_args()

    main(args.model_type)
