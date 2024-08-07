import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a YAML config file.")
    parser.add_argument("config_path", type=str, help="Path to the config YAML file.")
    args = parser.parse_args()

    config = load_config(args.config_path)
    print(config)
