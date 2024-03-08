import os
import pandas as pd

import torch
from torchvision.io import read_image


class ImageNet1K(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_name = f"{self.img_labels.iloc[idx, 0]}.png"
        img_path = os.path.join(self.img_dir, image_name)
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 6]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
