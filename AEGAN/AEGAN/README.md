# AEGAN — Autoencoder GAN for Pokémon Sprites

An Autoencoder Generative Adversarial Network (AEGAN) for generating novel
Pokémon sprites. The architecture jointly trains four networks:

- **Generator** `G : z → x` — maps a latent vector to an RGB image (96×96).
- **Encoder** `E : x → z` — maps an RGB image back to a latent vector.
- **DiscriminatorImage (Dx)** — classifies real vs. generated images.
- **DiscriminatorLatent (Dz)** — classifies real (sampled from N(0, I))
  vs. encoded latent vectors.

