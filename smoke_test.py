"""End-to-end smoke test for the PokeGAN reimplementation.

Verifies that:
  1. The DCGAN Generator/Discriminator have correct shapes.
  2. The AEGAN Generator/Encoder/Dx/Dz have correct shapes.
  3. A single end-to-end train step runs for both models without error.
  4. ImageFolder can load the scraped dataset and produce a correctly
     shaped batch.

Run:
    python smoke_test.py
"""

import os
import sys
import traceback

import torch
from torch import nn, optim


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    failures = []

    # ------------------------------------------------------------------ #
    section("1. DCGAN forward + 1 train step")
    # ------------------------------------------------------------------ #
    try:
        sys.path.insert(0, os.path.abspath("."))
        from pokegan import Generator as DCG, Discriminator as DCD, weights_init

        device = torch.device("cpu")
        G = DCG().to(device); G.apply(weights_init)
        D = DCD().to(device); D.apply(weights_init)

        z = torch.randn(4, 16, 1, 1)
        fake = G(z)
        assert fake.shape == (4, 3, 64, 64), f"G output {fake.shape}"
        print(f"  Generator: z {tuple(z.shape)} -> img {tuple(fake.shape)}  OK")

        logits = D(fake)
        assert logits.shape == (4, 1), f"D output {logits.shape}"
        print(f"  Discriminator: img -> {tuple(logits.shape)}  OK")

        # One real-vs-fake train step
        crit = nn.BCELoss()
        optD = optim.Adam(D.parameters(), lr=2.8e-4, betas=(0.5, 0.999))
        optG = optim.Adam(G.parameters(), lr=2.8e-4, betas=(0.5, 0.999))

        real = torch.rand(4, 3, 64, 64) * 2 - 1
        D.zero_grad()
        errD = crit(D(real).view(-1), torch.ones(4)) + \
               crit(D(G(z).detach()).view(-1), torch.zeros(4))
        errD.backward(); optD.step()

        G.zero_grad()
        errG = crit(D(G(z)).view(-1), torch.ones(4))
        errG.backward(); optG.step()
        print(f"  Train step: errD={errD.item():.4f}, errG={errG.item():.4f}  OK")
    except Exception:
        traceback.print_exc(); failures.append("DCGAN")

    # ------------------------------------------------------------------ #
    section("2. AEGAN forward + 1 train step")
    # ------------------------------------------------------------------ #
    try:
        sys.path.insert(0, os.path.abspath("AEGAN/AEGAN"))
        from aegan import (
            Generator as AG, Encoder as AE,
            DiscriminatorImage as DX, DiscriminatorLatent as DZ, AEGAN,
        )

        device = "cpu"
        latent_dim = 16
        bsz = 4

        Gn = AG(latent_dim=latent_dim).to(device)
        En = AE(latent_dim=latent_dim, device=device).to(device)
        Dx = DX(device=device).to(device)
        Dz = DZ(latent_dim=latent_dim, device=device).to(device)

        z = torch.randn(bsz, latent_dim)
        img = Gn(z)
        assert img.shape == (bsz, 3, 96, 96), f"G output {img.shape}"
        print(f"  Generator: z {tuple(z.shape)} -> img {tuple(img.shape)}  OK")

        z_hat = En(img)
        assert z_hat.shape == (bsz, latent_dim), f"E output {z_hat.shape}"
        print(f"  Encoder:   img -> z {tuple(z_hat.shape)}  OK")

        dx_out = Dx(img)
        dz_out = Dz(z)
        assert dx_out.shape == (bsz, 1) and dz_out.shape == (bsz, 1)
        print(f"  Dx: {tuple(dx_out.shape)}, Dz: {tuple(dz_out.shape)}  OK")

        # One full AEGAN step using a tiny in-memory loader.
        class _Dummy(torch.utils.data.Dataset):
            def __len__(self): return 16
            def __getitem__(self, _): return torch.rand(3, 96, 96) * 2 - 1, 0
        dl = torch.utils.data.DataLoader(_Dummy(), batch_size=bsz, drop_last=True)

        noise_fn = lambda n: torch.randn(n, latent_dim, device=device)
        gan = AEGAN(latent_dim, noise_fn, dl, batch_size=bsz, device=device)
        real = next(iter(dl))[0]
        ldx, ldz = gan.train_step_discriminators(real)
        lgx, lgz, lrx, lrz = gan.train_step_generators(real)
        print(f"  Train step: Dx={ldx:.3f} Dz={ldz:.3f} "
              f"Gx={lgx:.3f} Gz={lgz:.3f} Rx={lrx:.3f} Rz={lrz:.3f}  OK")
    except Exception:
        traceback.print_exc(); failures.append("AEGAN")

    # ------------------------------------------------------------------ #
    section("3. ImageFolder on scraped dataset")
    # ------------------------------------------------------------------ #
    try:
        import torchvision.transforms as T
        from torchvision import datasets

        path = "input/"
        if not os.path.isdir("input/pokemon_dataset"):
            print("  SKIP: no input/pokemon_dataset (run scrape.py first)")
        else:
            n = len([f for f in os.listdir("input/pokemon_dataset")
                     if f.endswith(".jpg")])
            print(f"  Found {n} JPEGs in input/pokemon_dataset/")

            tfm = T.Compose([
                T.Resize(64), T.CenterCrop(64), T.ToTensor(),
                T.Normalize((0.5,)*3, (0.5,)*3),
            ])
            ds = datasets.ImageFolder(root=path, transform=tfm)
            dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
            batch, _ = next(iter(dl))
            assert batch.shape == (8, 3, 64, 64), f"batch {batch.shape}"
            print(f"  ImageFolder -> batch {tuple(batch.shape)}, "
                  f"min={batch.min():.3f} max={batch.max():.3f}  OK")
    except Exception:
        traceback.print_exc(); failures.append("ImageFolder")

    # ------------------------------------------------------------------ #
    section("Summary")
    # ------------------------------------------------------------------ #
    if failures:
        print(f"  FAILED: {failures}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
