# jax-diffusion-lm

A minimal, self-contained implementation of a **Discrete Masked Diffusion Language Model (MDLM)** and the **Muon optimizer** written from scratch in pure JAX.

No Flax. No Haiku. No Optax. No PyTorch. No HuggingFace.

Just functional JAX, explicit PyTree parameters, and XLA compilation (`jax.jit`).

---

## Why this exists

Most diffusion codebases are either written for continuous 2D image latents (DDPM, Stable Diffusion) or buried under heavy framework abstractions.

This repository implements discrete text diffusion from first principles:

1. **Discrete Masked Diffusion over continuous Gaussian noise**: Continuous diffusion on text requires projecting Gaussian vectors back onto discrete token embeddings ("rounding"), which often collapses representation spaces. Masked diffusion instead defines a continuous-time Markov chain directly on discrete states, using `[MASK]` as an absorbing state.
2. **Bidirectional context instead of causal masking**: Autoregressive models (GPT) generate strictly left-to-right using triangular masks. Masked diffusion models evaluate all visible tokens bidirectionally, learning to infill missing spans arbitrarily.
3. **Muon optimizer from scratch**: Implements Keller Jordan's Newton-Schulz iteration (polar decomposition) to update 2D weight matrices with orthogonalized momentum, paired with AdamW for 1D vectors and embedding tables.

---

## Core Mechanics

### 1. Forward Corruption & Loss
Given a clean sequence $x_0 \in \mathcal{V}^L$ and continuous time $t \in (0, 1]$:

$$q(x_t \mid x_0) = \prod_{i=1}^L \left( (1 - t)\delta(x_t^i, x_0^i) + t \delta(x_t^i, [\text{MASK}]) \right)$$

Each token independently transitions to `[MASK]` with probability $t$. The variational lower bound (ELBO) simplifies to categorical cross-entropy computed **strictly on the corrupted positions**:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_t} \left[ \frac{1}{\sum_i \mathbb{I}(x_t^i = [\text{MASK}])} \sum_{i: x_t^i = [\text{MASK}]} -\log p_\theta(x_0^i \mid x_t, t) \right]$$

Unmasked tokens already reveal the ground truth; calculating loss on them would degrade training into trivial identity copying.

### 2. Reverse Sampling (Denoising)
Generation starts at $t = 1.0$ from an array of 100% `[MASK]` tokens. For each step transitioning from $t$ down to $t_{\text{next}}$:

1. Forward pass predicts $\hat{x}_0 \sim \text{Categorical}(\text{logits} / \tau)$.
2. Each currently masked position unmasks with probability:
   
$$P(\text{unmask}) = \frac{t - t_{\text{next}}}{t}$$

3. Tokens that do not unmask remain `[MASK]` to be resolved in later steps with richer bidirectional context.

### 3. Muon: Momentum Orthogonalized by Newton-Schulz
Standard optimizers allow singular values of weight matrices to grow unevenly. Muon computes the orthogonal polar factor $O = M(M^T M)^{-1/2}$ of momentum matrix $M$ using 5 iterations of a quintic polynomial:

$$X_0 = \frac{M}{\|M\|_F + \epsilon}, \quad X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k$$

with coefficients $a = 3.4445, b = -4.7750, c = 2.0315$.

- **2D weight matrices** (Attention projections, MLP weights) are updated with Muon.
- **1D parameters & Embeddings** (Biases, LayerNorm scales, Embedding tables) are updated with AdamW.

---

## Quickstart

### Setup
```bash
pip install -r requirements.txt
```

*(Requires `jax`, `jaxlib`, and `numpy`. Runs out-of-the-box on CPU, Apple Silicon GPU via Metal, or CUDA).*

### 1. Reverse Diffusion Demo
Visualize the step-by-step unmasking process (from pure noise to text):
```bash
python sample.py
```

Output:
```text
Step [00/10] (t=1.00 - 100% Noise):
  ████████████████████████████████████████████████████████████████
Step [02/10] (t=0.80 | Remaining masks: 49 - 76.6%):
  █n████████<█BV██████a████F████████sC███&███A█B█████>(█o█r███████
Step [05/10] (t=0.50 | Remaining masks: 31 - 48.4%):
  █ny█U██)█]<█BV3██t]█a███FI██r^██sC█\█&	o█A(B█O██g>(█o█r███████
Step [10/10] (t=0.00 | Remaining masks: 00 -  0.0%):
  %ny\U6()[]<RBV3"<t]*ag_)FIUKr^}OsCL\e&	oGA(BRO@\g>(aofr+c~1ki
```

### 4. Semi-Autoregressive Block Diffusion
Full sequence diffusion is bound to a fixed sequence length $L$. Block Diffusion combines causal autoregression across blocks with bidirectional diffusion inside each block:

- **Across Blocks (Causal)**: Block $k$ attends strictly to previously generated, clean blocks $0, \dots, k-1$.
- **Within Block (Diffusion)**: Tokens inside block $k$ attend bidirectionally to each other, generated concurrently in $S$ reverse diffusion steps.
- **Block-Causal Mask**: Token $i$ attends to token $j$ iff $\lfloor j / B \rfloor \le \lfloor i / B \rfloor$.

This enables streaming generation of arbitrary sequence lengths with speculative parallel decoding.

---

## Quickstart

### Setup
```bash
pip install -r requirements.txt
```

### 1. Reverse Diffusion Demo
Visualize full-sequence unmasking (from pure noise to text):
```bash
python sample.py
```

### 2. Block Diffusion Demo
Generate text block-by-block using semi-autoregressive diffusion:
```bash
python block_sample.py
```

Output:
```text
  [Block 0]: "?7 (=:9fQj#BHZFh"
  [Block 1]: "c0)r~iDYXnmoXq1z"
  [Block 2]: "rq,-|exg*EWin_"
  [Block 3]: "*u<aGHog4pfdVSyD"
```

### 3. Train
Train the model on a toy corpus with fused `jax.jit` compilation:
```bash
python train.py
```

---

## Taxonomy of Diffusion Language Models

| Family | Forward Process | Space | Key Works | Strengths & Trade-offs |
|---|---|---|---|---|
| **Continuous / Latent Diffusion** | Gaussian noise $\mathcal{N}(0, \sigma^2 I)$ | $\mathbb{R}^D$ embeddings | Diffusion-LM, CDCD, Plaid | Continuous guidance, but suffers from "rounding error" back to discrete tokens |
| **Categorical / State-Space Diffusion** | Uniform discrete noise transitions | $\mathcal{V}$ discrete states | D3PM (uniform) | Exact discrete formulation; slower convergence than absorbing state |
| **Masked Diffusion (MDLM)** | Absorbing state $[ \text{MASK} ]$ with prob $t$ | $\mathcal{V} \cup \{ [ \text{MASK} ] \}$ | D3PM absorbing, MDLM | Strongest empirical discrete baseline; ELBO reduces cleanly to cross-entropy |
| **Score Entropy Diffusion** | Concrete score matching on jump processes | Discrete score ratios | SEDD (ICML 2024) | Scalable likelihood matching GPT-2 without token rounding |
| **Block / Semi-Autoregressive Diffusion** | Causal across blocks, diffusion within | Block-partitioned | BD3PM, Block-Diffusion | Solves fixed-length bottleneck, supports streaming and speculative decoding |

---

## File Structure

```text
.
├── train.py                  # Training loop with jax.value_and_grad and jax.jit
├── sample.py                 # Full sequence step-by-step reverse sampling demo
├── block_sample.py           # Semi-autoregressive block diffusion demo
├── requirements.txt          # Minimal dependencies
└── src/
    ├── model.py              # Bidirectional Transformer with Block-Causal attention
    ├── diffusion.py          # Forward corruption, ELBO loss, and reverse sampling
    ├── block_diffusion.py    # Block-causal mask, block corruption, and block sampler
    ├── tokenizer.py          # Minimal character tokenizer
    ├── utils.py              # PyTree inspection, parameter counting, batching
    └── optimizers/
        ├── muon.py           # Pure JAX Muon (Newton-Schulz) + AdamW hybrid
        └── schedulers.py     # Branchless Cosine Warmup scheduler
```

---

## References

- **MDLM**: Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (2024).
- **D3PM**: Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces* (NeurIPS 2021).
- **SEDD**: Lou et al., *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (ICML 2024).
- **Block-Diffusion**: Arora et al., *Block-State Diffusion for Semi-Autoregressive Sequence Modeling* (2024).
- **Muon**: Keller Jordan et al., *Muon: An optimizer for hidden layers in neural networks* (2024), implemented in [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
