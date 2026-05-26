import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from train_hw4 import PromptIRModel


def pad_to_multiple(tensor, base=16):
    """Pad (1, C, H, W) tensor so H and W are divisible by base."""
    _, _, h, w = tensor.shape
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, h, w


def restore_image(model, inp, tta=False):
    """Run model, optionally with horizontal flip TTA."""
    out = model(inp)
    if tta:
        out_flip = model(torch.flip(inp, dims=[-1]))
        out = (out + torch.flip(out_flip, dims=[-1])) / 2
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test/degraded/ directory')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to .ckpt checkpoint file')
    parser.add_argument('--output', type=str, default='pred.npz',
                        help='Output file name (must be pred.npz inside zip)')
    parser.add_argument('--tta', action='store_true',
                        help='Enable horizontal flip test-time augmentation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = PromptIRModel.load_from_checkpoint(args.ckpt)
    model.eval().to(device)

    to_tensor = ToTensor()
    test_dir = Path(args.test_dir)

    img_files = sorted(
        [f for f in test_dir.iterdir()
         if f.suffix.lower() in ('.png', '.jpg', '.jpeg')],
        key=lambda f: int(f.stem),
    )
    print(f'Found {len(img_files)} test images')

    images_dict = {}

    with torch.no_grad():
        for img_path in tqdm(img_files, desc='Restoring'):
            img_np = np.array(Image.open(img_path).convert('RGB'))

            inp = to_tensor(img_np).unsqueeze(0).to(device)
            inp, orig_h, orig_w = pad_to_multiple(inp, base=16)

            out = restore_image(model, inp, tta=args.tta)

            out = out[:, :, :orig_h, :orig_w]
            out = torch.clamp(out, 0, 1)
            out_np = (out.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

            images_dict[img_path.name] = out_np

    assert len(images_dict) == 100, \
        f'Expected 100 images, got {len(images_dict)}'
    for k, v in images_dict.items():
        assert v.shape[0] == 3, f'{k}: expected shape (3, H, W), got {v.shape}'
        assert v.dtype == np.uint8, f'{k}: expected uint8, got {v.dtype}'

    np.savez(args.output, **images_dict)
    print(f'Saved {len(images_dict)} images to {args.output}')
    print('Remember to zip this file before uploading to CodaBench!')


if __name__ == '__main__':
    main()
