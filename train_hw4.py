import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from dataset_hw4 import RainSnowDataset


def compute_psnr(pred, target):
    pred = torch.clamp(pred, 0, 1)
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse.item())


class PromptIRModel(pl.LightningModule):
    def __init__(self, lr=2e-4, warmup_epochs=15, max_epochs=150):
        super().__init__()
        self.save_hyperparameters()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([_, de_id], degrad, clean) = batch
        restored = self.net(degrad)
        loss = self.loss_fn(restored, clean)
        self.log('train_loss', loss,
                 prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ([_, de_id], degrad, clean) = batch
        restored = self.net(degrad)
        loss = self.loss_fn(restored, clean)
        psnr = compute_psnr(restored, clean)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_psnr', psnr, prog_bar=True, on_epoch=True)
        return psnr

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return [optimizer], [scheduler]


def main():
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to train/ directory (contains degraded/ and clean/)')
    parser.add_argument('--ckpt_dir', type=str, default='train_ckpt_hw4')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    train_set = RainSnowDataset(
        args.data_dir, patch_size=args.patch_size,
        is_train=True, val_ratio=args.val_ratio,
    )
    val_set = RainSnowDataset(
        args.data_dir, patch_size=args.patch_size,
        is_train=False, val_ratio=args.val_ratio,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = PromptIRModel(
        lr=args.lr,
        warmup_epochs=min(15, args.epochs // 10),
        max_epochs=args.epochs,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='{epoch:03d}-{val_psnr:.2f}',
        monitor='val_psnr',
        mode='max',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
    )

    logger = CSVLogger(save_dir='logs_hw4/', name='promptir')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[ckpt_callback],
        precision='16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
