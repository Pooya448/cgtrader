import torch
from torch.utils.data import Dataset, DataLoader
import deepdish as dd
from pathlib import Path
import os
import numpy as np


class ShapeNetDataset(Dataset):
    def __init__(self, args):
        self.dataset_path = Path(args["dataset_path"])
        self.class_code = args["class_code"]
        self.files = [
            file
            for file in self.dataset_path.glob("*.dd")
            if self.class_code in file.stem
        ]
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        self.shuffle = args["shuffle"]

        # print("Dataset size: ", len(self.files))
        print(f"Found {len(self.files)} files for class {self.class_code}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        voxel_data = dd.io.load(str(file_path))["data"]
        voxel_tensor = torch.tensor(voxel_data, dtype=torch.float32).unsqueeze(0)
        voxel_tensor = voxel_tensor * 0.5 + 0.5
        return {"voxels": voxel_tensor}

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def get_loader(self, shuffle=True):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            worker_init_fn=self.worker_init_fn,
            drop_last=True,
        )


if __name__ == "__main__":
    # Example usage
    import yaml

    config = {
        "dataset_path": "data/",
        "class_code": "chair",
        "batch_size": 32,
        "num_workers": 4,
        "shuffle": True,
    }
    dataset = ShapeNetDataset(config)
    print(f'Found {len(dataset)} files for class {config["class_code"]}.')
