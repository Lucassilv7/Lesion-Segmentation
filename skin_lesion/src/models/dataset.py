import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SkinLesionDataset(Dataset):
    """
    Dataset do ISIC 2018 para segmentação de lesões dermatoscópicas.

    Args:
        image_dir:  Pasta com as imagens pré-processadas.
        mask_dir:   Pasta com as máscaras de ground truth.
        transform:  Pipeline de augmentation do albumentations (opcional).
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        filename = self.images[index]

        img_path = self.image_dir / filename
        mask_path = self.mask_dir / filename
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask 