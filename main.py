import argparse
from config.parse import load_config
from trainers.TrainerVAE import TrainerVAE
import wandb


def main(args):
    config_path = f"config/{args.model_type.lower()}.yaml"
    config = load_config(config_path)

    wandb.init(project="3DGen-Task", config=config, name=args.run_name)
    config = wandb.config

    if args.model_type.lower() == "vae":
        trainer = TrainerVAE(config, args.run_name)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    trainer.run()

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The type of model to use (e.g., VAE)",
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="The name of the run"
    )
    args = parser.parse_args()

    main(args)
