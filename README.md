#### PokeGAN (wip)


1. **DCGAN** ([`pokegan.py`](./pokegan.py)) — a Deep Convolutional GAN,
   64 × 64 RGB outputs, trained with `DistributedDataParallel`.
2. **AEGAN** ([`AEGAN/AEGAN/`](./AEGAN/AEGAN)) — an Autoencoder GAN with four
   networks (Generator + Encoder + image- and latent-discriminators),
   96 × 96 RGB outputs.

#### Project Structure

```
pokeGAN/
├── pokegan.py                  # DCGAN training script
├── job.sh                      # SLURM job script for DCGAN
├── env.sh                      # (env setup placeholder)
├── input/                      # Training images (ImageFolder layout)
│   └── pokemon_dataset/
│       └── *.jpg
├── Models/                     # Saved DCGAN weights
├── generatedtest/              # Per-epoch DCGAN sample grids
├── Result/                     # GPU profiling CSV/result logs
└── AEGAN/
    └── AEGAN/
        ├── aegan.py            # All AEGAN model classes
        ├── main.py             # AEGAN training entry point
        ├── test.py             # AEGAN inference script
        ├── requirements.txt    # AEGAN dependencies
        ├── job.sh              # SLURM job script for AEGAN
        └── pause.json          # Runtime pause control
```

Run:

```bash
pip install torch torchvision matplotlib seaborn numpy pillow
python pokegan.py --ex test
```

#### DCGAN Hyperparameters

| Parameter | Value |
|---|---|
| Latent dim | 16 |
| Image size | 64 × 64 |
| Batch size | 128 |
| Epochs | 500 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Learning rate (G and D) | 0.00028 |
| Loss | BCELoss |
| Parallelism | `DataParallel` + `DistributedDataParallel` (gloo) |

Per-epoch image grids are saved to `generated<ex>/generated-images-NNNN.png`,
final weights to `<ex>G.pth` / `<ex>D.pth`, a loss plot to `<ex>GDLoss.png`,
and a training animation to `<ex>animation.gif`.

