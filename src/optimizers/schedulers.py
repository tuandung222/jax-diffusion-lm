"""
================================================================================
LEARNING RATE SCHEDULER: LINEAR WARMUP WITH COSINE DECAY (PURE JAX)
================================================================================
A 100% pure JAX functional implementation of the cosine warmup schedule.
Fully stateless, pure, and compatible with the XLA compiler (`jax.jit`).

WHY WARMUP + COSINE ANNEALING IN TRANSFORMERS & DIFFUSION MODELS?
-----------------------------------------------------------------
1. LINEAR WARMUP PHASE:
   - At initialization, Transformer weights are randomly distributed, and early
     gradients exhibit high variance and noise.
   - Applying a large peak learning rate immediately can cause catastrophic updates
     that knock the model into poorly conditioned parameter regions.
   - Linear warmup starts at 0 and ramps up to `base_lr` over `warmup_steps`,
     allowing the model's momentum buffers and internal statistics to stabilize.

2. COSINE DECAY PHASE:
   - Following warmup, smoothly decreasing the learning rate along a half-cosine
     curve enables fine-grained convergence into deep local minima without oscillation.
   - The cosine function provides zero-derivative transitions at both the peak and
     trough, avoiding abrupt gradient shocks compared to linear or step decay.

WHY `jnp.where` INSTEAD OF PYTHON `if/else`?
--------------------------------------------
JAX traces computational graphs to compile XLA primitives. Standard Python branching
(`if step < warmup_steps:`) evaluates at trace time and fails when `step` is a dynamic
JAX array inside `@jax.jit`. `jnp.where` provides branchless, hardware-accelerated
selection directly on accelerators (GPU/Metal/TPU).
"""

import jax
import jax.numpy as jnp


def cosine_warmup_schedule(
    step: jnp.ndarray,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6
) -> jnp.ndarray:
    """
    Computes the learning rate at a given `step` using Linear Warmup + Cosine Decay.

    MATHEMATICAL FORMULATION:
    -------------------------
    1. Warmup Phase (step < warmup_steps):
         lr(step) = base_lr * (step / warmup_steps)

    2. Cosine Decay Phase (step >= warmup_steps):
         progress = (step - warmup_steps) / (total_steps - warmup_steps)
         lr(step) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))

    Args:
        step: Current training step (scalar int or float `jnp.ndarray`).
        base_lr: Peak learning rate reached at the end of the warmup phase.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at the end of training.

    Returns:
        Learning rate as a scalar `jnp.ndarray`.
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    warmup_steps = float(warmup_steps)
    total_steps = float(total_steps)

    # 1. Linear warmup: ramp up from 0 -> base_lr
    # Use jnp.maximum(1.0, warmup_steps) to prevent division by zero if warmup_steps is 0
    warmup_lr = base_lr * (step / jnp.maximum(1.0, warmup_steps))

    # 2. Cosine decay: smooth decay from base_lr -> min_lr
    decay_steps = jnp.maximum(1.0, total_steps - warmup_steps)
    # Clip progress to strictly remain in [0.0, 1.0]
    progress = jnp.clip((step - warmup_steps) / decay_steps, 0.0, 1.0)
    cosine_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + jnp.cos(jnp.pi * progress))

    # 3. Branchless select for JIT compilation
    return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)
