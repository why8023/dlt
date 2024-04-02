import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ECG_PPG_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sample_files = list(self.data_dir.glob("feats_*.npy"))
        self.target_files = list(self.data_dir.glob("targets_*.npy"))

        self.sample_maps = []
        self.target_maps = []
        self.sample_indices = []

        for i, (sample_file, target_file) in enumerate(
            zip(self.sample_files, self.target_files)
        ):
            sample_map = np.load(sample_file, mmap_mode="r")
            target_map = np.load(target_file, mmap_mode="r")
            logging.info(f"load {sample_file} and {target_file}")

            self.sample_maps.append(sample_map)
            self.target_maps.append(target_map)

            for j in range(len(sample_map)):
                self.sample_indices.append((i, j))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        file_index, sample_index = self.sample_indices[idx]
        sample = self.sample_maps[file_index][sample_index].copy()
        sample = sample.reshape(1, -1)
        target = self.target_maps[file_index][sample_index].copy()

        if self.transform:
            sample = self.transform.augment(sample)

        return torch.from_numpy(sample).float(), torch.from_numpy(target).float()


if __name__ == "__main__":
    # 示例用法
    root_directory = "data"
    custom_dataset = ECG_PPG_Dataset(data_dir=root_directory)

    # 访问第一个样本和标签
    first_sample, first_target = custom_dataset[0]
    print(first_sample, first_target)
