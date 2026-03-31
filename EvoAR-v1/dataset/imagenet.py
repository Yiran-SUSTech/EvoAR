import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = "flip" in self.feature_dir

        aug_feature_dir = feature_dir.replace("ten_crop/", "ten_crop_105/")
        aug_label_dir = label_dir.replace("ten_crop/", "ten_crop_105/")
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files)
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1).item() < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir

        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]
        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))

        prefix_valid_mask = torch.ones((1,), dtype=torch.bool)
        valid = torch.tensor(1, dtype=torch.long)
        return torch.from_numpy(features), torch.from_numpy(labels), prefix_valid_mask, valid


def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), (
        "please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    )
    return CustomDataset(feature_dir, label_dir)
