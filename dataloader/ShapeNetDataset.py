import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
            if self.class_code == "All" or self.class_code in file.stem
        ]
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        self.shuffle = args["shuffle"]
        self.split_ratio = args["split_ratio"]

        num_files = len(self.files)
        split = int(np.floor(self.split_ratio * num_files))
        self.train_files, self.test_files = random_split(
            self.files, [split, num_files - split]
        )

        print(f"Found {len(self.files)} files for class {self.class_code}.")
        print(
            f"Training set size: {len(self.train_files)}, Testing set size: {len(self.test_files)}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        voxel_data = dd.io.load(str(file_path))["data"]
        voxel_tensor = torch.tensor(voxel_data, dtype=torch.float32)
        voxel_tensor = torch.permute(voxel_tensor, (2, 0, 1)).unsqueeze(0)
        voxel_tensor = voxel_tensor * 0.5 + 0.5
        return {"voxels": voxel_tensor}

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def get_loader(self, train=True):
        files = self.train_files if train else self.test_files
        dataset = torch.utils.data.Subset(self, files.indices)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle if train else False,
            worker_init_fn=self.worker_init_fn,
            drop_last=True,
        )


if __name__ == "__main__":
    # Example test

    config = {
        "dataset_path": "data/",
        "class_code": None,
        "batch_size": 32,
        "num_workers": 4,
        "shuffle": True,
        "split_ratio": 0.8,
    }
    dataset = ShapeNetDataset(config)
    print(f'Found {len(dataset)} files for class {config["class_code"]}.')
    train_loader = dataset.get_loader(train=True)
    test_loader = dataset.get_loader(train=False)
    print(
        f"Training set size: {len(train_loader.dataset)}, Testing set size: {len(test_loader.dataset)}"
    )
