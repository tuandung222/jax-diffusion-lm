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
6. Multi-dataset support: Built-in Toy Corpus, Tiny Shakespeare, or custom local .txt.
"""

import os
import sys
import time
import math
import argparse
import urllib.request
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


def load_training_text(dataset: str, data_path: str = None) -> str:
    """
    Loads raw training text from a custom path, Tiny Shakespeare, or toy corpus.
    """
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Custom data file not found: {data_path}")
        print(f"Loading custom dataset from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    if dataset == "shakespeare":
        os.makedirs("data", exist_ok=True)
        shakespeare_path = os.path.join("data", "tinyshakespeare.txt")
        if not os.path.exists(shakespeare_path):
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(f"Downloading Tiny Shakespeare dataset from {url}...")
            urllib.request.urlretrieve(url, shakespeare_path)
            print(f"Saved to {shakespeare_path}")
        
        with open(shakespeare_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded Tiny Shakespeare ({len(text):,} characters).")
        return text

    # Default: Toy corpus
    print("Using built-in educational toy corpus.")
    return TOY_CORPUS * 50


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pure JAX Discrete Diffusion LM with Muon")
    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "shakespeare"],
                        help="Pre-configured dataset to use ('toy' or 'shakespeare')")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to custom text file (overrides --dataset)")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length in characters")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--steps", type=int, default=250, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=25, help="Warmup steps")
    parser.add_argument("--lr", type=float, default=2e-3, help="Peak learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--sample_every", type=int, default=50, help="Steps between generation samples")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 75)
    print(" STARTING PURE JAX DIFFUSION LM TRAINING WITH MUON OPTIMIZER")
    print("=" * 75)

    # 1. Device detection and master PRNG key initialization
    devices = jax.devices()
    print(f"JAX Computational Devices: {devices}")
    
    master_key = jax.random.PRNGKey(2026)
    master_key, k_init, k_train = jax.random.split(master_key, 3)

    # 2. Load text dataset
    raw_text = load_training_text(args.dataset, args.data_path)

    # 3. Tokenizer initialization and dataset chunking
    tokenizer = CharTokenizer([raw_text])
    print(f"Character Vocabulary Size: {tokenizer.vocab_size} tokens")
    print(f"Special Tokens: MASK_ID={tokenizer.mask_id}, PAD_ID={tokenizer.pad_id}")

    dataset = prepare_dataset(raw_text, tokenizer, seq_len=args.seq_len)
    num_samples = dataset.shape[0]
    print(f"Prepared {num_samples} training samples of fixed length L={args.seq_len} characters.")

    # 4. Model Parameters
    print("\n--- Initializing Pure JAX Model Parameters ---")
    params = init_transformer_params(
        k_init,
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_model * 4,
        max_len=args.seq_len
    )
    print_model_summary(params)

    # 5. Initialize Muon + AdamW hybrid optimizer state
    print("\n--- Initializing Muon Optimizer State (Newton-Schulz) ---")
    opt_state = init_muon_state(params)
    print(f"Initial optimizer state step: {opt_state.step}")

    # ==========================================================================
    # PURE FUNCTIONAL JIT-COMPILED TRAINING STEP
    # ==========================================================================
    @jax.jit
    def train_step(params, opt_state, batch, rng, lr):
        """
        Executes a single atomic training step.
        """
        def loss_wrapper(p):
            return compute_loss(
                p,
                rng=rng,
                x_0=batch,
                mask_token_id=tokenizer.mask_id,
                pad_token_id=tokenizer.pad_id,
                num_heads=args.num_heads
            )

        (loss, metrics), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)

        new_params, new_opt_state = muon_step(
            params=params,
            grads=grads,
            state=opt_state,
            lr=lr,
            weight_decay=0.01,
            ns_steps=5
        )

        return new_params, new_opt_state, metrics

    # 6. Main Training Loop
    print("\n--- Starting Training Loop ---")
    start_time = time.time()
    
    num_batches = num_samples // args.batch_size
    if num_batches == 0:
        raise ValueError(f"Dataset has {num_samples} samples, smaller than batch_size {args.batch_size}!")

    for step in range(1, args.steps + 1):
        k_train, k_step, k_sample = jax.random.split(k_train, 3)

        # Cyclic batch indexing
        batch_idx = (step - 1) % num_batches
        batch = dataset[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

        # Compute dynamic learning rate
        current_lr = cosine_warmup_schedule(
            step=step,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            total_steps=args.steps
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
                f"Step [{step:03d}/{args.steps:03d}] | "
                f"Loss: {loss_val:.4f} | "
                f"Masked Acc: {acc_val:5.1f}% | "
                f"Mask Ratio: {ratio_val:4.1f}% | "
                f"LR: {float(current_lr):.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # Intermediate sampling
        if step % args.sample_every == 0:
            print("\n" + "-" * 60)
            print(f" INTERMEDIATE DIFFUSION SAMPLING AT STEP {step}:")
            print("-" * 60)
            generated_ids = sample_tokens(
                params=params,
                rng=k_sample,
                batch_size=1,
                seq_len=args.seq_len,
                num_steps=32,
                mask_token_id=tokenizer.mask_id,
                num_heads=args.num_heads,
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
