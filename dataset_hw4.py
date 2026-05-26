import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.image_utils import crop_img, random_augmentation


class RainSnowDataset(Dataset):
    DE_RAIN = 0
    DE_SNOW = 1

    def __init__(self, data_root, patch_size=128, is_train=True,
                 val_ratio=0.05, seed=42):
        super().__init__()
        self.patch_size = patch_size
        self.is_train = is_train
        self.to_tensor = ToTensor()

        deg_dir = Path(data_root) / 'degraded'
        cln_dir = Path(data_root) / 'clean'

        pairs = []
        for p in sorted(deg_dir.glob('rain-*.png')):
            idx = p.stem.split('-')[1]
            c = cln_dir / f'rain_clean-{idx}.png'
            if c.exists():
                pairs.append((str(p), str(c), self.DE_RAIN))

        for p in sorted(deg_dir.glob('snow-*.png')):
            idx = p.stem.split('-')[1]
            c = cln_dir / f'snow_clean-{idx}.png'
            if c.exists():
                pairs.append((str(p), str(c), self.DE_SNOW))

        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))

        self.pairs = pairs[n_val:] if is_train else pairs[:n_val]
        split = 'Train' if is_train else 'Val'
        print(f'{split} set: {len(self.pairs)} pairs')

    def _random_crop(self, img1, img2):
        H, W = img1.shape[:2]
        ps = self.patch_size
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        return img1[top:top + ps, left:left + ps], img2[top:top + ps, left:left + ps]

    def __getitem__(self, idx):
        deg_path, cln_path, de_type = self.pairs[idx]

        deg_img = crop_img(np.array(Image.open(deg_path).convert('RGB')), base=16)
        cln_img = crop_img(np.array(Image.open(cln_path).convert('RGB')), base=16)

        if self.is_train:
            deg_patch, cln_patch = self._random_crop(deg_img, cln_img)
            deg_patch, cln_patch = random_augmentation(deg_patch, cln_patch)
        else:
            deg_patch, cln_patch = deg_img, cln_img

        name = Path(deg_path).stem
        return [name, de_type], self.to_tensor(deg_patch), self.to_tensor(cln_patch)

    def __len__(self):
        return len(self.pairs)
