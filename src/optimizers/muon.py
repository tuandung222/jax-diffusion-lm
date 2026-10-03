"""
================================================================================
OPTIMIZER MODULE: MUON (MOMENTUM ORTHOGONALIZED BY NEWTON-SCHULZ) IN PURE JAX
================================================================================
A 100% pure JAX implementation of the Muon optimizer, with zero dependencies
on Flax, Haiku, or Optax.
Algorithm originally proposed by Keller Jordan (2024) and popularized in modded-nanogpt.

WHY USE MUON FOR TRANSFORMER MATRICES?
--------------------------------------
In modern deep neural networks (Transformers, ConvNets, Diffusion Backbones):
1. 2D weight matrices (Linear projections, Attention QKV, MLP weights) represent
   the vast majority of trainable parameters.
2. Standard SGD with momentum or AdamW often produce ill-conditioned weight matrices;
   a few dominant singular directions can capture disproportionate spectral energy,
   leading to gradient spikes and slow convergence.
3. Muon resolves this by:
   - Accumulating standard momentum: M_t = beta * M_{t-1} + (1 - beta) * G_t
   - Orthogonalizing the momentum matrix via Polar Decomposition before updating:
       O_t = M_t * (M_t^T * M_t)^(-1/2)
     such that all non-zero singular values of the update matrix are normalized to 1.
   - This distributes update energy uniformly across all subspace dimensions,
     delivering 2-3x faster empirical convergence compared to vanilla AdamW.

WHY NEWTON-SCHULZ ITERATION INSTEAD OF EXACT SVD?
-------------------------------------------------
Computing exact Singular Value Decomposition (SVD) has an O(m * n^2) complexity
and is notoriously slow and difficult to parallelize on hardware accelerators (GPU/TPU/Metal).
The Newton-Schulz polynomial iteration uses only General Matrix Multiplications (GEMM) -
the exact operation accelerators are engineered to perform at maximum throughput.
"""

from typing import NamedTuple, Any, Tuple, Dict, Union
import jax
import jax.numpy as jnp


def zeropower_via_newtonschulz5(
    G: jnp.ndarray,
    steps: int = 5,
    eps: float = 1e-7
) -> jnp.ndarray:
    """
    Computes the approximate orthogonal polar factor O = G * (G^T G)^(-1/2)
    via quintic (5th-order) Newton-Schulz polynomial iterations.

    MATHEMATICAL INTUITION:
    -----------------------
    Let G have singular value decomposition G = U * Sigma * V^T.
    The closest orthogonal matrix to G in the Frobenius norm is O = U * V^T.
    To map all singular values Sigma -> I without explicitly calculating U and V:
    
      X_0 = G / (||G||_F + eps)  (Normalizes spectral radius into the convergence basin)
      X_{k+1} = a * X_k + b * (X_k X_k^T) * X_k + c * (X_k X_k^T)^2 * X_k

    Optimal quintic coefficients (empirically derived by Keller Jordan):
      a = 3.4445
      b = -4.7750
      c = 2.0315
    The polynomial f(s) = a*s + b*s^3 + c*s^5 maps singular values s in [0, 1]
    sharply toward 1.0 within only 5 iterations.

    SHAPE-AWARE MATRIX MULTIPLICATION:
    ----------------------------------
    If m > n (more rows than columns), computing X * X^T produces a large (m, m) matrix.
    We transpose G -> G^T so that inner square products are always min(m, n) x min(m, n),
    conserving substantial VRAM and execution cycles.

    Args:
        G: 2D matrix of shape (m, n).
        steps: Number of polynomial iterations (5 is standard for convergence).
        eps: Epsilon for numerical stability during Frobenius normalization.

    Returns:
        Orthogonalized matrix of the same shape (m, n).
    """
    assert G.ndim == 2, f"Newton-Schulz requires a 2D matrix, received shape: {G.shape}"
    m, n = G.shape
    transposed = False

    # Ensure m <= n so that the inner product X @ X^T is (m, m), the minimal dimension
    if m > n:
        G = G.T
        m, n = n, m
        transposed = True

    # Step 1: Frobenius normalization to guarantee spectral radius is in the basin of convergence
    norm = jnp.linalg.norm(G) + eps
    X = G / norm

    # Quintic iteration coefficients
    a = 3.4445
    b = -4.7750
    c = 2.0315

    # Step 2: Quintic iterations consisting purely of GEMM operations
    for _ in range(steps):
        A = jnp.matmul(X, X.T)       # Shape (m, m)
        AA = jnp.matmul(A, A)         # Shape (m, m)
        B = b * A + c * AA            # Shape (m, m)
        X = a * X + jnp.matmul(B, X)  # Shape (m, n)

    # Transpose back if originally m > n
    if transposed:
        X = X.T

    return X


# ==============================================================================
# PYTREE STATE MANAGEMENT IN PURE JAX
# ==============================================================================
# In JAX, functions must remain purely functional (no in-place mutations or hidden states).
# Instead of storing states in an object attribute (`self.state`), optimizer states
# are organized as a PyTree of NamedTuples mirrored against the model parameter tree.

class LeafMuonState(NamedTuple):
    """Stores the momentum buffer for a 2D weight matrix optimized with Muon."""
    momentum: jnp.ndarray


class LeafAdamWState(NamedTuple):
    """Stores first-moment (m) and second-moment (v) buffers for AdamW parameters."""
    m: jnp.ndarray
    v: jnp.ndarray


class MuonOptState(NamedTuple):
    """
    Global hybrid optimizer state:
    - step: Integer scalar tracking the current optimization step count.
    - leaf_states: A PyTree matching the model's `params` structure, containing
      LeafMuonState or LeafAdamWState at each leaf node.
    """
    step: jnp.ndarray
    leaf_states: Any


def default_is_muon_leaf(path: Tuple[Any, ...], p: jnp.ndarray) -> bool:
    """
    Predicate function partitioning parameters into Muon vs. AdamW:
    
    1. USE MUON:
       - Must be a 2D weight matrix (p.ndim == 2).
       - Must not be an Embedding table or final Output Head (since token embeddings
         are sparse lookup tables where orthogonalization assumptions do not hold).
       - Examples: Attention QKV projection, Output projection, MLP weights.

    2. USE ADAMW (Fallback):
       - 1D parameters: Biases, LayerNorm/RMSNorm scale and shift.
       - Token & Positional Embedding tables.
       - Language Model Head (vocabulary projection matrix).
    """
    path_str = "/".join(str(getattr(k, "key", k)) for k in path).lower()
    
    # Exclude embeddings and heads from Muon
    if "embed" in path_str or "head" in path_str or "wte" in path_str or "wpe" in path_str:
        return False
        
    return p.ndim == 2


def init_muon_state(params: Any, is_muon_fn=default_is_muon_leaf) -> MuonOptState:
    """
    Initializes the optimizer state PyTree matching the model parameters.

    Args:
        params: Model parameter PyTree (nested dict/list of jnp.ndarray).
        is_muon_fn: Predicate callable checking whether a leaf qualifies for Muon.

    Returns:
        MuonOptState containing the step counter and initialized state PyTree.
    """
    def _init_leaf(path, p):
        if is_muon_fn(path, p):
            return LeafMuonState(momentum=jnp.zeros_like(p))
        else:
            return LeafAdamWState(
                m=jnp.zeros_like(p),
                v=jnp.zeros_like(p)
            )

    leaf_states = jax.tree_util.tree_map_with_path(_init_leaf, params)
    return MuonOptState(step=jnp.array(0, dtype=jnp.int32), leaf_states=leaf_states)


def muon_step(
    params: Any,
    grads: Any,
    state: MuonOptState,
    lr: Union[float, jnp.ndarray],
    muon_momentum: float = 0.95,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.99,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.01,
    ns_steps: int = 5,
) -> Tuple[Any, MuonOptState]:
    """
    Executes a single optimization update step.
    Fully functional and 100% compatible with `jax.jit`.

    ALGORITHM WORKFLOW:
    -------------------
    1. For Muon parameters (2D weights):
       - Momentum update: M_t = beta * M_{t-1} + (1 - beta) * G_t
       - Orthogonal polar update: O_t = zeropower_via_newtonschulz5(M_t)
       - Aspect ratio scaling: scale = max(1.0, sqrt(m / n))
         (Preserves relative magnitude across rectangular matrices)
       - Weight decay and parameter step:
         W_{t+1} = W_t * (1 - lr * wd) - lr * scale * O_t

    2. For AdamW parameters (1D parameters & Embeddings):
       - 1st moment: m_t = beta1 * m_{t-1} + (1 - beta1) * G_t
       - 2nd moment: v_t = beta2 * v_{t-1} + (1 - beta2) * G_t^2
       - Bias correction:
         m_hat = m_t / (1 - beta1^t),  v_hat = v_t / (1 - beta2^t)
       - Update: theta_{t+1} = theta_t * (1 - lr * wd) - lr * (m_hat / (sqrt(v_hat) + eps))

    Returns:
        (new_params, new_state): Tuple of updated parameters and new optimizer state.
    """
    step = state.step + 1
    lr = jnp.asarray(lr, dtype=jnp.float32)

    def _update_leaf(p: jnp.ndarray, g: jnp.ndarray, s: Any) -> Tuple[jnp.ndarray, Any]:
        if isinstance(s, LeafMuonState):
            # --- Branch 1: Muon Update ---
            # 1.1 Accumulate momentum
            new_mom = muon_momentum * s.momentum + (1.0 - muon_momentum) * g
            
            # 1.2 Orthogonalize momentum via Newton-Schulz
            ortho_update = zeropower_via_newtonschulz5(new_mom, steps=ns_steps)
            
            # 1.3 Aspect-ratio preservation scale
            m, n = p.shape
            scale = jnp.maximum(1.0, jnp.sqrt(float(m) / float(n)))
            
            # 1.4 Decoupled weight decay and update
            decayed_p = p * (1.0 - lr * weight_decay)
            new_p = decayed_p - lr * scale * ortho_update
            
            return new_p, LeafMuonState(momentum=new_mom)

        elif isinstance(s, LeafAdamWState):
            # --- Branch 2: AdamW Fallback Update ---
            # 2.1 Accumulate 1st and 2nd moments
            new_m = adam_beta1 * s.m + (1.0 - adam_beta1) * g
            new_v = adam_beta2 * s.v + (1.0 - adam_beta2) * (g ** 2)

            # 2.2 Bias correction
            bc1 = 1.0 - adam_beta1 ** step
            bc2 = 1.0 - adam_beta2 ** step
            m_hat = new_m / bc1
            v_hat = new_v / bc2

            update = m_hat / (jnp.sqrt(v_hat) + adam_eps)
            
            # 2.3 Weight decay only applied to >= 2D tensors (e.g. embeddings), skipped for biases/norms
            wd = weight_decay if p.ndim >= 2 else 0.0
            decayed_p = p * (1.0 - lr * wd)
            new_p = decayed_p - lr * update

            return new_p, LeafAdamWState(m=new_m, v=new_v)
        else:
            raise TypeError(f"Invalid optimizer leaf state type: {type(s)}")

    # Flatten PyTrees to iterate across matching (param, grad, state) tuples
    flat_params, tree_def = jax.tree_util.tree_flatten(params)
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    flat_states, state_tree_def = jax.tree_util.tree_flatten(
        state.leaf_states,
        is_leaf=lambda x: isinstance(x, (LeafMuonState, LeafAdamWState))
    )

    new_params_flat = []
    new_leaf_states_flat = []

    for p, g, s in zip(flat_params, flat_grads, flat_states):
        np, ns = _update_leaf(p, g, s)
        new_params_flat.append(np)
        new_leaf_states_flat.append(ns)

    # Reconstruct original hierarchical PyTree structure
    new_params = jax.tree_util.tree_unflatten(tree_def, new_params_flat)
    new_leaf_states = jax.tree_util.tree_unflatten(state_tree_def, new_leaf_states_flat)

    new_state = MuonOptState(step=step, leaf_states=new_leaf_states)
    return new_params, new_state
