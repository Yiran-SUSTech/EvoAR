import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Text2ImgDataset(Dataset):
    def __init__(self, args, transform):
        img_path_list = []
        valid_file_paths = []
        for lst_name in sorted(os.listdir(args.data_path)):
            if lst_name.endswith(".jsonl"):
                valid_file_paths.append(os.path.join(args.data_path, lst_name))

        for file_path in valid_file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data["image_path"]
                    code_dir = os.path.splitext(os.path.basename(file_path))[0]
                    img_path_list.append((img_path, code_dir, line_idx))

        self.img_path_list = img_path_list
        self.transform = transform
        self.t5_feat_path = args.t5_feat_path
        self.short_t5_feat_path = args.short_t5_feat_path
        self.t5_feat_path_base = os.path.basename(self.t5_feat_path)
        self.short_t5_feat_path_base = (
            os.path.basename(self.short_t5_feat_path)
            if self.short_t5_feat_path is not None
            else self.t5_feat_path_base
        )
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = args.cls_token_num
        self.t5_feature_dim = 2048

    def __len__(self):
        return len(self.img_path_list)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim), dtype=torch.float32)
        prefix_valid_mask = torch.zeros((self.t5_feature_max_len,), dtype=torch.bool)
        valid = torch.tensor(0, dtype=torch.long)
        return img, t5_feat_padding, prefix_valid_mask, valid

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return self.dummy_data()

        if min(img.size) < self.image_size:
            return self.dummy_data()

        if self.transform is not None:
            img = self.transform(img)

        t5_file = os.path.join(self.t5_feat_path, code_dir, f"{code_name}.npy")
        if self.short_t5_feat_path is not None and torch.rand(1).item() < 0.3:
            t5_file = t5_file.replace(self.t5_feat_path_base, self.short_t5_feat_path_base)

        if not os.path.isfile(t5_file):
            return self.dummy_data()

        try:
            t5_feat = torch.from_numpy(np.load(t5_file)).float()
        except Exception:
            return self.dummy_data()

        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim), dtype=torch.float32)
        prefix_valid_mask = torch.zeros((self.t5_feature_max_len,), dtype=torch.bool)

        t5_feat_len = t5_feat.shape[1]
        feat_len = min(self.t5_feature_max_len, t5_feat_len)
        t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
        prefix_valid_mask[-feat_len:] = True
        valid = torch.tensor(1, dtype=torch.long)
        return img, t5_feat_padding, prefix_valid_mask, valid


def build_t2i(args, transform):
    return Text2ImgDataset(args, transform)
