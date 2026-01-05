"""
DCGAN for CIFAR-10 (32x32 RGB).

Usage:
  python3 cute_gan.py --epochs 50

Outputs:
  - Sample image grids: ./outputs/epoch_0001.png ...
  - Checkpoints: ./checkpoints/ckpt-*
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf


@dataclass(frozen=True)
class GanConfig:
    latent_dim: int = 128
    batch_size: int = 128
    epochs: int = 50
    lr: float = 2e-4
    beta_1: float = 0.5
    sample_every: int = 1
    num_sample_images: int = 64
    seed: int = 42
    outdir: str = "outputs"
    checkpoint_dir: str = "checkpoints"


def load_cifar10(batch_size: int) -> tf.data.Dataset:
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)
    # Scale to [-1, 1] for tanh generator output
    x_train = (x_train / 127.5) - 1.0
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(buffer_size=min(50_000, len(x_train)), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


def make_generator(latent_dim: int) -> tf.keras.Model:
    # 32x32x3 output
    inputs = tf.keras.Input(shape=(latent_dim,), name="z")
    x = tf.keras.layers.Dense(4 * 4 * 256, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((4, 4, 256))(x)

    x = tf.keras.layers.Conv2DTranspose(
        128, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)  # 8x8
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
        64, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)  # 16x16
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(
        32, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)  # 32x32
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv2D(
        3, kernel_size=3, strides=1, padding="same", activation="tanh", name="img"
    )(x)
    return tf.keras.Model(inputs, outputs, name="generator")


def make_discriminator() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(32, 32, 3), name="img")
    x = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    logits = tf.keras.layers.Dense(1, name="logits")(x)  # from_logits=True
    return tf.keras.Model(inputs, logits, name="discriminator")


def _to_uint8(images: np.ndarray) -> np.ndarray:
    # images: [-1,1] float32 -> [0,255] uint8
    images = (images + 1.0) * 127.5
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images


def save_image_grid(images: np.ndarray, path: Path, grid: tuple[int, int] | None = None) -> None:
    """
    images: (N, H, W, C) in [-1,1] float or [0,255] uint8.
    """
    if images.dtype != np.uint8:
        images = _to_uint8(images)

    n, h, w, c = images.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels, got {c}")

    if grid is None:
        cols = int(math.sqrt(n))
        cols = max(cols, 1)
        rows = int(math.ceil(n / cols))
    else:
        rows, cols = grid
        if rows * cols < n:
            raise ValueError("Grid too small for number of images")

    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx in range(n):
        r = idx // cols
        col = idx % cols
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w, :] = images[idx]

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


class DCGANTrainer:
    def __init__(self, cfg: GanConfig):
        self.cfg = cfg
        tf.random.set_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.generator = make_generator(cfg.latent_dim)
        self.discriminator = make_discriminator()

        self.g_opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr, beta_1=cfg.beta_1)
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr, beta_1=cfg.beta_1)

        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.fixed_noise = tf.random.normal([cfg.num_sample_images, cfg.latent_dim], seed=cfg.seed)

        ckpt = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            g_opt=self.g_opt,
            d_opt=self.d_opt,
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt, directory=cfg.checkpoint_dir, max_to_keep=5
        )

    def restore_latest(self) -> str | None:
        latest = self.ckpt_manager.latest_checkpoint
        if latest:
            self.ckpt_manager.checkpoint.restore(latest).expect_partial()
            return latest
        return None

    def d_loss(self, real_logits: tf.Tensor, fake_logits: tf.Tensor) -> tf.Tensor:
        real_labels = tf.ones_like(real_logits)
        fake_labels = tf.zeros_like(fake_logits)
        return self.bce(real_labels, real_logits) + self.bce(fake_labels, fake_logits)

    def g_loss(self, fake_logits: tf.Tensor) -> tf.Tensor:
        # Want discriminator to classify fakes as real
        real_labels = tf.ones_like(fake_logits)
        return self.bce(real_labels, fake_logits)

    @tf.function
    def train_step(self, real_images: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        noise = tf.random.normal([self.cfg.batch_size, self.cfg.latent_dim])

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake_images = self.generator(noise, training=True)

            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)

            d_loss = self.d_loss(real_logits, fake_logits)
            g_loss = self.g_loss(fake_logits)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        return d_loss, g_loss

    def sample_and_save(self, epoch: int) -> Path:
        imgs = self.generator(self.fixed_noise, training=False).numpy()
        out_path = Path(self.cfg.outdir) / f"epoch_{epoch:04d}.png"
        save_image_grid(imgs, out_path)
        return out_path

    def train(self, dataset: tf.data.Dataset) -> None:
        restored = self.restore_latest()
        if restored:
            print(f"Restored checkpoint: {restored}")

        for epoch in range(1, self.cfg.epochs + 1):
            d_losses = []
            g_losses = []
            for real_batch in tqdm(dataset, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False):
                d_loss, g_loss = self.train_step(real_batch)
                d_losses.append(float(d_loss.numpy()))
                g_losses.append(float(g_loss.numpy()))

            print(
                f"Epoch {epoch:04d}: "
                f"D_loss={np.mean(d_losses):.4f}  G_loss={np.mean(g_losses):.4f}"
            )

            if self.cfg.sample_every > 0 and (epoch % self.cfg.sample_every == 0):
                p = self.sample_and_save(epoch)
                print(f"Saved samples: {p}")

            ckpt_path = self.ckpt_manager.save()
            print(f"Saved checkpoint: {ckpt_path}")


def parse_args() -> GanConfig:
    p = argparse.ArgumentParser(description="DCGAN on CIFAR-10 (TensorFlow/Keras).")
    p.add_argument("--latent-dim", type=int, default=GanConfig.latent_dim)
    p.add_argument("--batch-size", type=int, default=GanConfig.batch_size)
    p.add_argument("--epochs", type=int, default=GanConfig.epochs)
    p.add_argument("--lr", type=float, default=GanConfig.lr)
    p.add_argument("--beta-1", type=float, default=GanConfig.beta_1)
    p.add_argument("--sample-every", type=int, default=GanConfig.sample_every)
    p.add_argument("--num-sample-images", type=int, default=GanConfig.num_sample_images)
    p.add_argument("--seed", type=int, default=GanConfig.seed)
    p.add_argument("--outdir", type=str, default=GanConfig.outdir)
    p.add_argument("--checkpoint-dir", type=str, default=GanConfig.checkpoint_dir)
    args = p.parse_args()
    return GanConfig(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta_1=args.beta_1,
        sample_every=args.sample_every,
        num_sample_images=args.num_sample_images,
        seed=args.seed,
        outdir=args.outdir,
        checkpoint_dir=args.checkpoint_dir,
    )


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    dataset = load_cifar10(cfg.batch_size)
    trainer = DCGANTrainer(cfg)
    trainer.train(dataset)


if __name__ == "__main__":
    main()

