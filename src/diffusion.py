"""
================================================================================
DIFFUSION PROCESS: DISCRETE MASKED DIFFUSION (MDLM / ABSORBING STATE)
================================================================================
A 100% pure JAX implementation of Discrete Masked Diffusion for language modeling,
inspired by modern formulations including D3PM, Mask-Predict, and MDLM (2024).

CONTINUOUS DIFFUSION (IMAGES) VS. DISCRETE DIFFUSION (LANGUAGE):
---------------------------------------------------------------
1. CONTINUOUS DIFFUSION (e.g. DDPM, Stable Diffusion):
   - Pixel space is continuous (R^D).
   - The forward process gradually adds Gaussian noise:
       x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * eps
   - The network is trained to predict the noise vector eps.

2. DISCRETE DIFFUSION (Language Models):
   - Language consists of discrete tokens belonging to a finite vocabulary V.
   - Adding Gaussian noise to token embeddings requires "rounding" back to discrete
     vocabulary entries, which suffers from severe rounding errors and representation collapse.
   - Masked Diffusion solves this by defining the Markov chain directly on the discrete
     state space with an Absorbing State (`<mask`>):
     * Forward Noising: At timestep t in [0, 1], each token independently transitions
       to the `<mask`> token with probability p = t.
     * At t = 0: Clean sequence x_0 (0% noise).
     * At t = 1: Sequence is 100% `<mask`> tokens (maximum entropy).
     * Reverse Sampling: Starting from all `<mask`>, the network predicts clean token
       distributions and iteratively unmasks positions over discrete time intervals.

ADVANTAGES OF MASKED DIFFUSION OVER AUTOREGRESSIVE LMs (GPT):
------------------------------------------------------------
- Bidirectional Infilling: Can condition on arbitrary fixed tokens and infill missing spans
  anywhere in the sequence (prefix, middle, suffix).
- Flexible Generation Speed: Generation time can be varied dynamically by choosing the number
  of diffusion steps (e.g. 10 to 64 steps) rather than strictly requiring L sequential steps.
"""

from typing import Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
from .model import forward_transformer


def q_sample(
    rng: jax.random.PRNGKey,
    x_0: jnp.ndarray,
    t: jnp.ndarray,
    mask_token_id: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward corruption process q(x_t | x_0, t): introduces noise via token masking.

    MECHANISM:
    ----------
    For each sequence in the batch at time t in [0, 1], an independent Bernoulli
    random variable is sampled for each token with success probability p = t.
    Tokens sampled as True are replaced with `mask_token_id`.

    Args:
        rng: JAX PRNGKey.
        x_0: Clean token IDs of shape (batch_size, seq_len).
        t: 1D array of continuous timesteps for each sequence, shape (batch_size,).
        mask_token_id: Integer ID of the `<mask`> token.

    Returns:
        x_t: Corrupted token array of shape (batch_size, seq_len).
        mask_indices: Boolean array of shape (batch_size, seq_len), True where masked.
    """
    # Expand timestep dimensions for sequence broadcasting: (B, 1)
    p_mask = t[:, None]
    
    # Independent Bernoulli draw for every token in the batch
    mask_indices = jax.random.bernoulli(rng, p=p_mask, shape=x_0.shape)
    
    # Replace masked positions with mask_token_id
    x_t = jnp.where(mask_indices, mask_token_id, x_0)
    
    return x_t, mask_indices


def compute_loss(
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    x_0: jnp.ndarray,
    mask_token_id: int,
    pad_token_id: int = 0,
    num_heads: int = 8,
    min_t: float = 1e-4
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Computes the Evidence Lower Bound (ELBO) Cross-Entropy training loss.

    MATHEMATICAL FORMULATION:
    -------------------------
    In Discrete Masked Diffusion, the variational lower bound reduces to categorical
    cross-entropy evaluated specifically on masked token positions:
    
      L(theta) = E_{t ~ U(0, 1), x_t ~ q(x_t|x_0)} [ -log p_theta(x_0 | x_t, t) ]

    KEY IMPLEMENTATION DETAILS:
    - Loss is evaluated ONLY on positions that were corrupted (`mask_indices == True`).
      Unmasked positions already reveal the ground truth; computing loss on them would
      cause the model to learn trivial identity copying.
    - `<pad>` tokens are excluded from loss accumulation.

    Args:
        params: Model parameter PyTree.
        rng: JAX PRNGKey.
        x_0: Ground truth clean token array of shape (B, L).
        mask_token_id: `<mask`> token ID.
        pad_token_id: `<pad>` token ID to ignore in loss calculation.
        num_heads: Number of attention heads.
        min_t: Minimum timestep cutoff to prevent numerical degeneration at t=0.

    Returns:
        loss: Scalar cross-entropy loss for backpropagation.
        metrics: Dictionary containing loss, masked accuracy, and masking ratio.
    """
    B, L = x_0.shape
    rng_t, rng_noise = jax.random.split(rng)

    # 1. Sample continuous timesteps uniformly in [min_t, 1.0] for each sample in batch
    t = jax.random.uniform(rng_t, shape=(B,), minval=min_t, maxval=1.0)

    # 2. Corrupt clean sequences to construct x_t
    x_t, mask_indices = q_sample(rng_noise, x_0, t, mask_token_id)

    # 3. Model forward pass predicts logits over vocabulary
    logits = forward_transformer(params, x_t, t, num_heads=num_heads)  # (B, L, vocab_size)

    # 4. Numerically stable Cross-Entropy computation
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, L, vocab_size)
    
    # Extract log probability of the ground-truth target token
    target_log_probs = jnp.take_along_axis(log_probs, x_0[..., None], axis=-1).squeeze(-1)  # (B, L)

    # Valid loss mask: only include positions that are MASK and NOT PAD
    valid_mask = mask_indices & (x_0 != pad_token_id)
    num_valid = jnp.maximum(1.0, jnp.sum(valid_mask))
    
    # Mean cross-entropy over valid masked tokens
    loss = -jnp.sum(target_log_probs * valid_mask) / num_valid

    # 5. Monitor masked accuracy (percentage of correctly predicted masked tokens)
    pred_tokens = jnp.argmax(logits, axis=-1)
    correct = (pred_tokens == x_0) & valid_mask
    acc = jnp.sum(correct) / num_valid

    metrics = {
        "loss": loss,
        "masked_acc": acc,
        "masked_ratio": jnp.mean(mask_indices)
    }
    return loss, metrics


def sample_tokens(
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    mask_token_id: int,
    num_heads: int = 8,
    temperature: float = 1.0
) -> jnp.ndarray:
    """
    Ancestral Reverse Diffusion Sampling for text generation.

    GENERATION FROM PURE NOISE:
    ---------------------------
    1. Initialize at t = 1.0: Sequence consists entirely of `<mask`> tokens (100% noise).
    2. Divide interval [1.0 -> 0.0] into `num_steps` intervals: dt = 1.0 / num_steps.
    3. At each step transitioning from t down to t_next:
       - Pass current sequence x_t to Transformer to predict distribution over x_0.
       - Draw candidates from Categorical(logits / temperature).
       - Compute analytic unmask probability:
           p_unmask = (t - t_next) / t
       - For positions currently masked, unmask with probability p_unmask.
       - Remaining masked positions continue to be unmasked in subsequent steps,
         conditioned on newly revealed context!
    4. At the final step (t -> 0), unmask all remaining positions to finalize output.

    Args:
        params: Trained model parameter PyTree.
        rng: JAX PRNGKey.
        batch_size: Number of parallel sequences to generate.
        seq_len: Sequence length.
        num_steps: Number of reverse diffusion steps (e.g., 16, 32, 64).
        mask_token_id: Integer ID of `<mask`>.
        num_heads: Attention heads.
        temperature: Sampling temperature (higher = more diverse, lower = more focused).

    Returns:
        Array of generated token IDs of shape (batch_size, seq_len).
    """
    # Start with 100% mask tokens
    x = jnp.full((batch_size, seq_len), mask_token_id, dtype=jnp.int32)
    dt = 1.0 / float(num_steps)

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t_next_val = jnp.maximum(0.0, t_val - dt)
        t_arr = jnp.full((batch_size,), t_val, dtype=jnp.float32)

        rng, rng_step, rng_unmask = jax.random.split(rng, 3)

        # 1. Predict clean token distribution based on bidirectional visible context
        logits = forward_transformer(params, x, t_arr, num_heads=num_heads)
        
        # 2. Temperature scaling and categorical sampling
        scaled_logits = logits / jnp.maximum(1e-5, temperature)
        x_0_pred = jax.random.categorical(rng_step, scaled_logits, axis=-1)

        # 3. Determine which tokens to unmask at this step
        is_masked = (x == mask_token_id)
        if step == num_steps - 1:
            # Final step: unconditionally unmask all remaining positions
            unmask_prob = 1.0
        else:
            # Analytic transition probability
            unmask_prob = (t_val - t_next_val) / t_val

        unmask_flags = jax.random.bernoulli(rng_unmask, p=unmask_prob, shape=(batch_size, seq_len))
        should_unmask = is_masked & unmask_flags

        # 4. Update sequence: replace unmasked positions with predicted tokens
        x = jnp.where(should_unmask, x_0_pred, x)

    return x
