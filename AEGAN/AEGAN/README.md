# AEGAN — Autoencoder GAN for Pokémon Sprites

An Autoencoder Generative Adversarial Network (AEGAN) for generating novel
Pokémon sprites. The architecture jointly trains four networks:

- **Generator** `G : z → x` — maps a latent vector to an RGB image (96×96).
- **Encoder** `E : x → z` — maps an RGB image back to a latent vector.
- **DiscriminatorImage (Dx)** — classifies real vs. generated images.
- **DiscriminatorLatent (Dz)** — classifies real (sampled from N(0, I))
  vs. encoded latent vectors.

Two cycle-consistency reconstructions are enforced:

- `X_tilde = G(E(X))` — image reconstruction loss (L1).
- `Z_tilde = E(G(Z))` — latent reconstruction loss (MSE).

## Files

| File | Purpose |
|---|---|
| `aegan.py` | Model classes (`Generator`, `Encoder`, `DiscriminatorImage`, `DiscriminatorLatent`, `AEGAN`) |
| `main.py` | Training entry point |
| `test.py` | Inference / sampling from a saved generator |
| `requirements.txt` | Python dependencies |
| `job.sh` | SLURM submission script |
| `pause.json` | Runtime pause control — set `pause` > 0 to pause training |

## Dataset Layout

Place Pokémon sprite images inside `data/<subfolder>/` (any subfolder name
works — `ImageFolder` will treat it as a single class). For example:

```
data/
└── sprites_rgb/
    ├── img_0001.png
    ├── img_0002.png
    └── ...
```

## Training

```bash
pip install -r requirements.txt
python main.py
```

Outputs go to:

- `results/generated/gen.{epoch:04d}.png` — generated samples per epoch
- `results/reconstructed/gen.{epoch:04d}.png` — reconstructions per epoch
- `results/checkpoints/gen.{epoch:05d}.pt` — saved generator weights every 50 epochs

## Inference

```bash
python test.py
```

Loads `trained_generator_weights.pt` and saves 32 individual generated images.

## Hyperparameters

| Parameter | Value |
|---|---|
| Latent dimension | 16 |
| Image size | 96 × 96 |
| Batch size | 32 |
| Epochs | 500 |
| Generator / Encoder LR | 2e-4 |
| Discriminator LR | 1e-4 |
| Optimizer | Adam (β₁=0.5, β₂=0.999, wd=1e-8) |
