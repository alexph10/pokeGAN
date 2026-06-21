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

