"""
================================================================================
TRAINING SCRIPT: MINI DIFFUSION LANGUAGE MODEL WITH MUON OPTIMIZER (PURE JAX)
================================================================================
An end-to-end training pipeline combining all built-from-scratch components:
1. Bidirectional Transformer with Timestep Conditioning.
2. Discrete Masked Diffusion (MDLM) process.
3. Muon Optimizer (Newton-Schulz Polar Decomposition) with AdamW hybrid partitioning.
4. Cosine Warmup Learning Rate Scheduler.
5. High-performance fused compilation from Gradient to Parameter Update via `jax.jit`.

WHY `jax.value_and_grad` IS CLEANER FOR FIRST-PRINCIPLES LEARNING:
------------------------------------------------------------------
In PyTorch, calling `loss.backward()` invisibly accumulates gradients as side-effects
into `.grad` tensor attributes.
In JAX:
  `grad_fn = jax.grad(loss_fn)`
  `grads = grad_fn(params)`
Gradients are first-class, immutable PyTrees with the exact same layout as `params`.
Using `jax.value_and_grad(loss_fn, has_aux=True)` computes the scalar loss,
monitoring metrics (accuracy, masking ratio), and all partial gradients in a single
reverse-mode automatic differentiation pass.
"""

import time
import math
import jax
import jax.numpy as jnp

from src.tokenizer import CharTokenizer
from src.model import init_transformer_params
from src.diffusion import compute_loss, sample_tokens
from src.optimizers.muon import init_muon_state, muon_step
from src.optimizers.schedulers import cosine_warmup_schedule
from src.utils import print_model_summary, prepare_dataset


# ==============================================================================
# EDUCATIONAL TOY TRAINING CORPUS
# ==============================================================================
# Diverse text discussing AI, diffusion mathematics, and optimization
TOY_CORPUS = """
Artificial intelligence and deep learning have transformed the way we understand computation and cognition.
Diffusion models represent a paradigm shift in generative modeling, moving away from simple autoregressive next token prediction.
By learning to reverse a continuous or discrete corruption process, diffusion models can plan and generate coherent sequences bidirectionally.
The Muon optimizer leverages Newton Schulz iterations to compute orthogonal updates for large weight matrices.
Orthogonalized momentum maintains the spectral energy across all subspace directions, preventing gradient collapse.
Pure functional programming in JAX provides mathematical clarity, deterministic reproducibility, and accelerated execution through XLA compilation.
Language models understand grammar, syntax, semantics, and context by attending to bidirectional representations.
In masked diffusion models, tokens transition into an absorbing mask state during the forward noising process.
The neural network learns to denoise and recover the original words from corrupted observations.
Through iterative ancestral sampling, order emerges from pure noise step by step.
Linear warmup stabilizes the initial noisy gradients before cosine annealing guides convergence toward optimal parameter basins.
"""


def main():
    print("=" * 75)
    print(" STARTING PURE JAX DIFFUSION LM TRAINING WITH MUON OPTIMIZER")
    print("=" * 75)

    # 1. Device detection and master PRNG key initialization
    devices = jax.devices()
    print(f"JAX Computational Devices: {devices}")
    
    master_key = jax.random.PRNGKey(2026)
    master_key, k_init, k_train = jax.random.split(master_key, 3)

    # 2. Tokenizer initialization and dataset preparation
    tokenizer = CharTokenizer([TOY_CORPUS])
    print(f"Character Vocabulary Size: {tokenizer.vocab_size} tokens")
    print(f"Special Tokens: MASK_ID={tokenizer.mask_id}, PAD_ID={tokenizer.pad_id}")

    # Replicate text to create sufficient training samples
    full_text = TOY_CORPUS * 50
    seq_len = 64
    dataset = prepare_dataset(full_text, tokenizer, seq_len=seq_len)
    num_samples = dataset.shape[0]
    print(f"Prepared {num_samples} training samples of fixed length L={seq_len} characters.")

    # 3. Model Hyperparameters
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 512
    batch_size = 16
    total_steps = 250
    warmup_steps = 25
    base_lr = 2e-3
    weight_decay = 0.01

    print("\n--- Initializing Pure JAX Model Parameters ---")
    params = init_transformer_params(
        k_init,
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=seq_len
    )
    print_model_summary(params)

    # 4. Initialize Muon + AdamW hybrid optimizer state
    print("\n--- Initializing Muon Optimizer State (Newton-Schulz) ---")
    opt_state = init_muon_state(params)
    print(f"Initial optimizer state step: {opt_state.step}")

    # ==========================================================================
    # PURE FUNCTIONAL JIT-COMPILED TRAINING STEP
    # ==========================================================================
    # Forward Pass -> Loss Calculation -> Value and Grad -> Muon/AdamW Update
    # are fused into an optimized accelerator kernel by XLA.
    @jax.jit
    def train_step(params, opt_state, batch, rng, lr):
        """
        Executes a single atomic training step.
        """
        # Closure taking params as the sole differentiable argument
        def loss_wrapper(p):
            return compute_loss(
                p,
                rng=rng,
                x_0=batch,
                mask_token_id=tokenizer.mask_id,
                pad_token_id=tokenizer.pad_id,
                num_heads=num_heads
            )

        # Compute loss, metrics, and partial gradients simultaneously
        (loss, metrics), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)

        # Update parameters using the Muon optimizer
        new_params, new_opt_state = muon_step(
            params=params,
            grads=grads,
            state=opt_state,
            lr=lr,
            weight_decay=weight_decay,
            ns_steps=5
        )

        return new_params, new_opt_state, metrics

    # 5. Main Training Loop
    print("\n--- Starting Training Loop ---")
    start_time = time.time()
    
    num_batches = num_samples // batch_size

    for step in range(1, total_steps + 1):
        k_train, k_step, k_sample = jax.random.split(k_train, 3)

        # Fetch mini-batch
        batch_idx = (step - 1) % num_batches
        batch = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # Compute dynamic learning rate via Cosine Warmup Schedule
        current_lr = cosine_warmup_schedule(
            step=step,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        # Execute JIT-compiled optimization step
        params, opt_state, metrics = train_step(params, opt_state, batch, k_step, current_lr)

        # Periodic logging
        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start_time
            loss_val = float(metrics["loss"])
            acc_val = float(metrics["masked_acc"]) * 100.0
            ratio_val = float(metrics["masked_ratio"]) * 100.0
            print(
                f"Step [{step:03d}/{total_steps:03d}] | "
                f"Loss: {loss_val:.4f} | "
                f"Masked Acc: {acc_val:5.1f}% | "
                f"Mask Ratio: {ratio_val:4.1f}% | "
                f"LR: {float(current_lr):.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # 6. Showcase intermediate reverse diffusion sampling every 50 steps
        if step % 50 == 0:
            print("\n" + "-" * 60)
            print(f" INTERMEDIATE DIFFUSION SAMPLING AT STEP {step}:")
            print("-" * 60)
            generated_ids = sample_tokens(
                params=params,
                rng=k_sample,
                batch_size=1,
                seq_len=seq_len,
                num_steps=32,
                mask_token_id=tokenizer.mask_id,
                num_heads=num_heads,
                temperature=0.8
            )
            generated_text = tokenizer.decode(list(generated_ids[0]))
            print(f"Generated text sample:\n\"{generated_text}\"")
            print("-" * 60 + "\n")

    total_time = time.time() - start_time
    print("=" * 75)
    print(f" TRAINING COMPLETE! Total runtime: {total_time:.2f} seconds.")
    print("=" * 75)


if __name__ == "__main__":
    main()
