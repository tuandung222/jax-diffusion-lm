"""
================================================================================
BLOCK DIFFUSION MODULE: SEMI-AUTOREGRESSIVE BLOCK DIFFUSION (PURE JAX)
================================================================================
Implements Block Diffusion (Semi-Autoregressive Diffusion / BD3PM):
- Causal (autoregressive) dependency ACROSS blocks.
- Bidirectional non-autoregressive diffusion WITHIN each block.

WHY BLOCK DIFFUSION?
-------------------
Standard diffusion language models suffer from two primary limitations:
1. Fixed-Context Bottleneck: Models are trained with a fixed sequence length L
   and cannot easily stream or generate indefinitely long text.
2. Inference Cost at Long Lengths: Denoising an entire document at once in T steps
   requires quadratic attention across all tokens for every single step.

Block Diffusion solves both issues:
- Sequence length L is divided into blocks of size B (e.g. B = 16 or 32 tokens).
- Autoregressive Across Blocks: Block k attends causally to clean, previously generated
  blocks 0, 1, ..., k-1.
- Bidirectional Diffusion Within Block: Tokens within block k are generated concurrently
  via a fast S-step diffusion unmasking process (S << B).
- Streaming & Arbitrary Length: Generating a new block only requires conditioning on the
  prefix cache/context, allowing arbitrary-length continuation!
"""

from typing import Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
from .model import forward_transformer


def make_block_causal_mask(seq_len: int, block_size: int) -> jnp.ndarray:
    """
    Constructs a 2D boolean attention mask implementing Block-Causal connectivity.

    MASK MATRIX STRUCTURE:
    ----------------------
    Let position i belong to block b_i = i // block_size,
    and position j belong to block b_j = j // block_size.

    Token i can attend to token j if and only if:
      b_j <= b_i

    - If b_j < b_i: Past block (causal observation, fully clean context).
    - If b_j == b_i: Same block (full bidirectional self-attention within block!).
    - If b_j > b_i: Future block (strictly masked out).

    Args:
        seq_len: Total sequence length L.
        block_size: Number of tokens per block B.

    Returns:
        Boolean array of shape (seq_len, seq_len), True where attention is permitted.
    """
    pos = jnp.arange(seq_len)
    block_indices = pos // block_size  # (L,)
    
    # Broadcast comparison: block_j <= block_i
    # Shape: (L, L), where row is query (i) and column is key (j)
    mask = block_indices[None, :] <= block_indices[:, None]
    return mask


def block_q_sample(
    rng: jax.random.PRNGKey,
    x_0: jnp.ndarray,
    target_block: jnp.ndarray,
    t: jnp.ndarray,
    block_size: int,
    mask_token_id: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward corruption for Block Diffusion training:
    - Blocks < target_block: Clean tokens (x_0, noise level = 0).
    - Block == target_block: Corrupted with masking probability t.
    - Blocks > target_block: Masked out or ignored.

    Args:
        rng: JAX PRNGKey.
        x_0: Clean token IDs of shape (batch_size, seq_len).
        target_block: 1D array of target block indices of shape (batch_size,).
        t: 1D array of diffusion times in [0, 1] for the target block (batch_size,).
        block_size: Size of each block B.
        mask_token_id: Integer ID of `<mask`>.

    Returns:
        x_t: Corrupted sequence array of shape (batch_size, seq_len).
        target_mask: Boolean array of shape (batch_size, seq_len), True strictly for
                     the masked tokens in the active target block.
    """
    B, L = x_0.shape
    pos = jnp.arange(L)
    token_blocks = pos[None, :] // block_size  # (1, L)
    target_blocks_expanded = target_block[:, None]  # (B, 1)

    is_past = token_blocks < target_blocks_expanded
    is_target = token_blocks == target_blocks_expanded
    is_future = token_blocks > target_blocks_expanded

    # Sample corruption mask for the target block
    p_mask = t[:, None]  # (B, 1)
    noise_draw = jax.random.bernoulli(rng, p=p_mask, shape=x_0.shape)
    target_mask = is_target & noise_draw

    # Compose corrupted sequence:
    # - Past: Keep original clean token x_0
    # - Target: Mask if noise_draw is True, else keep x_0
    # - Future: Mask all future tokens
    x_t = jnp.where(is_past, x_0, jnp.where(target_mask | is_future, mask_token_id, x_0))

    return x_t, target_mask


def compute_block_loss(
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    x_0: jnp.ndarray,
    block_size: int,
    mask_token_id: int,
    pad_token_id: int = 0,
    num_heads: int = 8,
    min_t: float = 1e-4
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Computes cross-entropy loss for Block Diffusion on the active target block.

    Args:
        params: Model parameter PyTree.
        rng: JAX PRNGKey.
        x_0: Clean batch array of shape (B, L).
        block_size: Size of each block B.
        mask_token_id: `<mask`> token ID.
        pad_token_id: `<pad>` token ID to ignore.
        num_heads: Number of attention heads.
        min_t: Minimum timestep cutoff.

    Returns:
        loss: Scalar training loss.
        metrics: Dictionary of training metrics.
    """
    B, L = x_0.shape
    num_blocks = L // block_size
    rng_block, rng_t, rng_noise = jax.random.split(rng, 3)

    # 1. Sample a random target block k in [0, num_blocks - 1] for each sequence
    target_block = jax.random.randint(rng_block, shape=(B,), minval=0, maxval=num_blocks)

    # 2. Sample continuous timestep t for the target block
    t = jax.random.uniform(rng_t, shape=(B,), minval=min_t, maxval=1.0)

    # 3. Corrupt target block tokens
    x_t, target_mask = block_q_sample(rng_noise, x_0, target_block, t, block_size, mask_token_id)

    # 4. Generate Block-Causal attention mask (L, L)
    attn_mask = make_block_causal_mask(L, block_size)

    # 5. Forward pass through Transformer
    logits = forward_transformer(params, x_t, t, num_heads=num_heads, attn_mask=attn_mask)

    # 6. Cross-entropy loss strictly on masked positions of the target block
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, x_0[..., None], axis=-1).squeeze(-1)

    valid_mask = target_mask & (x_0 != pad_token_id)
    num_valid = jnp.maximum(1.0, jnp.sum(valid_mask))
    loss = -jnp.sum(target_log_probs * valid_mask) / num_valid

    # Accuracy metric on target block masked tokens
    pred_tokens = jnp.argmax(logits, axis=-1)
    correct = (pred_tokens == x_0) & valid_mask
    acc = jnp.sum(correct) / num_valid

    metrics = {
        "loss": loss,
        "masked_acc": acc,
        "target_masked_count": jnp.sum(valid_mask) / B
    }
    return loss, metrics


def sample_block_diffusion(
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    num_blocks: int,
    block_size: int,
    steps_per_block: int,
    mask_token_id: int,
    num_heads: int = 8,
    temperature: float = 0.8
) -> jnp.ndarray:
    """
    Semi-Autoregressive Block Diffusion Sampling:
    Iterates sequentially across blocks 0, 1, ..., num_blocks-1.
    Within each block, runs a fast S-step reverse diffusion process.

    Args:
        params: Trained model parameters.
        rng: PRNGKey.
        num_blocks: Number of blocks to generate.
        block_size: Number of tokens per block.
        steps_per_block: Reverse diffusion steps for each block (e.g. 8 or 16).
        mask_token_id: `<mask`> token ID.
        num_heads: Attention heads.
        temperature: Categorical sampling temperature.

    Returns:
        Generated token sequence array of shape (1, num_blocks * block_size).
    """
    total_len = num_blocks * block_size
    attn_mask = make_block_causal_mask(total_len, block_size)

    # Initialize full sequence as all [MASK]
    x = jnp.full((1, total_len), mask_token_id, dtype=jnp.int32)
    dt = 1.0 / float(steps_per_block)

    for block_idx in range(num_blocks):
        b_start = block_idx * block_size
        b_end = b_start + block_size

        for s in range(steps_per_block):
            t_val = 1.0 - s * dt
            t_next_val = jnp.maximum(0.0, t_val - dt)
            t_arr = jnp.array([t_val], dtype=jnp.float32)

            rng, rng_step, rng_unmask = jax.random.split(rng, 3)

            # Predict logits conditioned on clean prefix + noisy current block
            logits = forward_transformer(params, x, t_arr, num_heads=num_heads, attn_mask=attn_mask)
            
            scaled_logits = logits / jnp.maximum(1e-5, temperature)
            x_0_pred = jax.random.categorical(rng_step, scaled_logits, axis=-1)

            # Unmask probability within the current block
            if s == steps_per_block - 1:
                unmask_prob = 1.0
            else:
                unmask_prob = (t_val - t_next_val) / t_val

            # Only unmask tokens within the current active block [b_start:b_end]
            block_is_masked = (x[:, b_start:b_end] == mask_token_id)
            unmask_draw = jax.random.bernoulli(rng_unmask, p=unmask_prob, shape=(1, block_size))
            should_unmask = block_is_masked & unmask_draw

            # Update current block
            current_block = x[:, b_start:b_end]
            pred_block = x_0_pred[:, b_start:b_end]
            updated_block = jnp.where(should_unmask, pred_block, current_block)

            # Insert updated block back into the sequence
            x = x.at[:, b_start:b_end].set(updated_block)

    return x
