"""
================================================================================
BLOCK DIFFUSION DEMO: SEMI-AUTOREGRESSIVE BLOCK GENERATION
================================================================================
Demonstrates Block Diffusion generation:
- Sequences are divided into blocks (e.g., 4 blocks of 16 characters = 64 characters).
- Block 0 is generated via 8 diffusion steps.
- Block 1 is generated conditioned on Block 0 via 8 diffusion steps.
- Block 2 is generated conditioned on Blocks 0 & 1, and so forth!
"""

import time
import jax
import jax.numpy as jnp

from src.tokenizer import CharTokenizer
from src.model import init_transformer_params
from src.block_diffusion import (
    make_block_causal_mask,
    sample_block_diffusion,
    compute_block_loss
)


def demo_block_diffusion():
    print("=" * 75)
    print(" SEMI-AUTOREGRESSIVE BLOCK DIFFUSION DEMONSTRATION")
    print("=" * 75)

    tokenizer = CharTokenizer()
    rng = jax.random.PRNGKey(2026)
    rng_init, rng_mask, rng_sample = jax.random.split(rng, 3)

    num_blocks = 4
    block_size = 16
    total_len = num_blocks * block_size  # 64 characters
    steps_per_block = 8

    print(f"Total Length: {total_len} chars | Blocks: {num_blocks} | Block Size: {block_size} chars")
    print(f"Diffusion Steps Per Block: {steps_per_block}")

    # 1. Inspect Block-Causal Mask
    mask = make_block_causal_mask(seq_len=total_len, block_size=block_size)
    print(f"\nBlock-Causal Attention Mask Shape: {mask.shape}")
    print("Connectivity Matrix (First 4 tokens of Block 0 vs Block 1):")
    # Show small 8x8 excerpt
    excerpt = mask[:8, :8].astype(int)
    print(excerpt)

    # 2. Initialize Model
    params = init_transformer_params(
        rng_init,
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        max_len=total_len
    )

    # 3. Test Block Loss Computation
    dummy_x = jnp.zeros((2, total_len), dtype=jnp.int32)
    loss, metrics = compute_block_loss(
        params,
        rng=rng_sample,
        x_0=dummy_x,
        block_size=block_size,
        mask_token_id=tokenizer.mask_id,
        num_heads=4
    )
    print(f"\nBlock Diffusion Loss forward check: Loss = {float(loss):.4f}")

    # 4. Generate text block by block
    print("\n--- Running Semi-Autoregressive Block Generation ---")
    start = time.time()
    generated_ids = sample_block_diffusion(
        params,
        rng=rng_sample,
        num_blocks=num_blocks,
        block_size=block_size,
        steps_per_block=steps_per_block,
        mask_token_id=tokenizer.mask_id,
        num_heads=4,
        temperature=0.8
    )
    elapsed = time.time() - start

    decoded = tokenizer.decode(list(generated_ids[0]))
    print(f"Generated text ({len(decoded)} chars) in {elapsed:.2f}s:")
    
    # Display block-by-block breakdown
    for b in range(num_blocks):
        sub_ids = generated_ids[0, b * block_size : (b + 1) * block_size]
        sub_text = tokenizer.decode(list(sub_ids))
        print(f"  [Block {b}]: \"{sub_text}\"")

    print("\nFull Sequence: \"" + decoded + "\"")
    print("=" * 75)


if __name__ == "__main__":
    demo_block_diffusion()
