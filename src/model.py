"""
================================================================================
MODEL ARCHITECTURE: PURE JAX BIDIRECTIONAL DIFFUSION TRANSFORMER
================================================================================
A 100% pure JAX implementation of a Bidirectional Transformer backbone designed
specifically for Diffusion Language Models.
Zero high-level framework dependencies (no Flax, Haiku, Keras, or PyTorch).

CORE ARCHITECTURAL CHARACTERISTICS FOR DIFFUSION LM:
---------------------------------------------------
1. BIDIRECTIONAL SELF-ATTENTION:
   - In autoregressive causal models (e.g., GPT), attention masks strictly prevent
     tokens from attending to future positions (lower-triangular causal masking).
   - In Diffusion LMs, the input sequence x_t has corrupted/masked positions distributed
     throughout the text. To accurately predict a missing token in the middle of a sentence,
     the model MUST attend to both left context (prefix) and right context (suffix).
   - Consequently, full bidirectional attention (as in BERT/RoBERTa) is required.

2. TIMESTEP CONDITIONING:
   - The model must know the current noise level t in [0, 1] to adjust its predictions:
     * When t ~ 1.0 (heavy corruption): The model leans heavily on marginal prior distributions.
     * When t ~ 0.0 (light corruption): The model relies on sharp local contextual cues.
   - Continuous time t is mapped to a high-dimensional vector via Sinusoidal Timestep Embeddings,
     processed through a Time-MLP, and injected directly into each Transformer block.

3. STATELESS FUNCTIONAL DESIGN IN JAX:
   - No `class Model(nn.Module)`.
   - Weights and biases are organized as a standard Python dictionary PyTree (`params = {...}`).
   - Operations are expressed as pure functions: `init_transformer_params` and `forward_transformer`.
"""

from typing import Dict, Any, Tuple
import math
import jax
import jax.numpy as jnp


def sinusoidal_timestep_embedding(
    timesteps: jnp.ndarray,
    dim: int,
    max_period: int = 10000
) -> jnp.ndarray:
    """
    Encodes continuous timesteps t in [0, 1] into multi-frequency sinusoidal vectors.

    MATHEMATICAL INTUITION:
    -----------------------
    Following the positional encoding formulation from "Attention Is All You Need",
    we project scalar t across geometric frequency bands:
    
      omega_k = exp(-log(max_period) * k / (dim // 2))
      emb = [cos(1000 * t * omega), sin(1000 * t * omega)]

    This allows the neural network to differentiate fine-grained noise levels
    (e.g., distinguishing t = 0.01 from t = 0.02 with high fidelity).

    Args:
        timesteps: 1D array of diffusion times of shape (batch_size,).
        dim: Target embedding dimension (typically matching d_model).
        max_period: Maximum period controlling the lowest frequency band.

    Returns:
        2D array of sinusoidal embeddings of shape (batch_size, dim).
    """
    half = dim // 2
    # Geometrically spaced angular frequencies
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    # Scale t by 1000 to span the frequency spectrum
    args = timesteps[:, None] * freqs[None, :] * 1000.0  # Shape: (batch_size, half)
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)  # Shape: (batch_size, 2 * half)
    
    # Pad an extra zero if dimension is odd
    if dim % 2 == 1:
        embedding = jnp.pad(embedding, ((0, 0), (0, 1)))
    return embedding


def layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float = 1e-5
) -> jnp.ndarray:
    """
    Layer Normalization over the last hidden dimension.
    Formula: y = ((x - mean) / sqrt(var + eps)) * scale + bias
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / jnp.sqrt(var + eps)
    return normed * scale + bias


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Fast Gaussian Error Linear Unit (GELU) approximation.
    Standard in modern Transformer architectures (BERT, GPT-2).
    """
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3))))


def init_transformer_params(
    rng: jax.random.PRNGKey,
    vocab_size: int,
    d_model: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    d_ff: int = 1024,
    max_len: int = 128
) -> Dict[str, Any]:
    """
    Initializes all parameter tensors for the Transformer Diffusion Backbone.

    INITIALIZATION SCHEME:
    ----------------------
    - Token & Position Embeddings: Normal distribution N(0, 0.02^2).
    - Linear projection weights: Scaled Gaussian N(0, 1 / sqrt(fan_in)).
    - Biases: Zero initialization.
    - LayerNorm scales: Ones initialization.

    Args:
        rng: JAX immutable PRNGKey.
        vocab_size: Vocabulary cardinality.
        d_model: Hidden representation dimension.
        num_heads: Number of attention heads.
        num_layers: Number of stacked Transformer blocks.
        d_ff: Feed-forward expansion dimension (typically 4 * d_model).
        max_len: Maximum sequence length.

    Returns:
        PyTree dictionary containing all model parameters.
    """
    params = {}
    
    # 1. Token and Position Embedding tables
    rng, k_tok, k_pos = jax.random.split(rng, 3)
    params["wte"] = jax.random.normal(k_tok, (vocab_size, d_model)) * 0.02
    params["wpe"] = jax.random.normal(k_pos, (max_len, d_model)) * 0.02

    # 2. Timestep conditioning MLP
    # Maps sinusoidal embedding (d_model) -> 2*d_model -> d_model
    rng, k_t1, k_t2 = jax.random.split(rng, 3)
    params["time_mlp"] = {
        "w1": jax.random.normal(k_t1, (d_model, d_model * 2)) * 0.02,
        "b1": jnp.zeros(d_model * 2),
        "w2": jax.random.normal(k_t2, (d_model * 2, d_model)) * 0.02,
        "b2": jnp.zeros(d_model),
    }

    # 3. Stacked Transformer Blocks
    layers = []
    for _ in range(num_layers):
        rng, k_qkv, k_out, k_fc1, k_fc2 = jax.random.split(rng, 5)
        layer_params = {
            # Pre-Attention LayerNorm
            "ln1_scale": jnp.ones(d_model),
            "ln1_bias": jnp.zeros(d_model),
            # Fused Q, K, V projection: shape (d_model, 3 * d_model)
            "qkv_w": jax.random.normal(k_qkv, (d_model, 3 * d_model)) * (1.0 / math.sqrt(d_model)),
            "qkv_b": jnp.zeros(3 * d_model),
            # Output attention projection
            "out_w": jax.random.normal(k_out, (d_model, d_model)) * (1.0 / math.sqrt(d_model)),
            "out_b": jnp.zeros(d_model),
            # Pre-MLP LayerNorm
            "ln2_scale": jnp.ones(d_model),
            "ln2_bias": jnp.zeros(d_model),
            # Feed-Forward MLP
            "fc1_w": jax.random.normal(k_fc1, (d_model, d_ff)) * (1.0 / math.sqrt(d_model)),
            "fc1_b": jnp.zeros(d_ff),
            "fc2_w": jax.random.normal(k_fc2, (d_ff, d_model)) * (1.0 / math.sqrt(d_ff)),
            "fc2_b": jnp.zeros(d_model),
        }
        layers.append(layer_params)
    params["layers"] = layers

    # 4. Final LayerNorm & Unembedding Head
    rng, k_head = jax.random.split(rng)
    params["ln_f_scale"] = jnp.ones(d_model)
    params["ln_f_bias"] = jnp.zeros(d_model)
    params["head_w"] = jax.random.normal(k_head, (d_model, vocab_size)) * 0.02
    params["head_b"] = jnp.zeros(vocab_size)

    return params


def bidirectional_attention(
    x: jnp.ndarray,
    qkv_w: jnp.ndarray,
    qkv_b: jnp.ndarray,
    out_w: jnp.ndarray,
    out_b: jnp.ndarray,
    num_heads: int
) -> jnp.ndarray:
    """
    Multi-Head Bidirectional Self-Attention.

    TENSOR SHAPES & FLOW:
    ---------------------
    - Input x: (B, L, D) where B=Batch, L=SeqLen, D=HiddenDim.
    - Fused QKV Projection: (B, L, D) @ (D, 3*D) -> (B, L, 3*D).
    - Split into Q, K, V each of shape: (B, L, D).
    - Reshape & Transpose: (B, num_heads, L, head_dim) where head_dim = D // num_heads.
    - Scaled Dot-Product: Scores = (Q @ K^T) / sqrt(head_dim) -> (B, num_heads, L, L).
    - Softmax: Attention Weights -> (B, num_heads, L, L).
    - Context Aggregation: Weights @ V -> (B, num_heads, L, head_dim).
    - Concatenate & Output Projection: (B, L, D) @ (D, D) -> (B, L, D).
    """
    B, L, D = x.shape
    head_dim = D // num_heads

    # 1. Parallel Q, K, V linear projection in a single GEMM call
    qkv = jnp.matmul(x, qkv_w) + qkv_b                     # (B, L, 3 * D)
    q, k, v = jnp.split(qkv, 3, axis=-1)                  # 3 arrays of (B, L, D)

    # 2. Reshape and transpose across attention heads
    q = jnp.transpose(jnp.reshape(q, (B, L, num_heads, head_dim)), (0, 2, 1, 3))  # (B, H, L, d_k)
    k = jnp.transpose(jnp.reshape(k, (B, L, num_heads, head_dim)), (0, 2, 1, 3))  # (B, H, L, d_k)
    v = jnp.transpose(jnp.reshape(v, (B, L, num_heads, head_dim)), (0, 2, 1, 3))  # (B, H, L, d_k)

    # 3. Scaled Dot-Product Attention (No causal mask; unconstrained bidirectional view)
    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale  # (B, H, L, L)
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # 4. Multiply with Value vectors
    context = jnp.matmul(attn_weights, v)  # (B, H, L, d_k)
    context = jnp.transpose(context, (0, 2, 1, 3))
    context = jnp.reshape(context, (B, L, D))

    # 5. Output projection
    return jnp.matmul(context, out_w) + out_b


def forward_transformer(
    params: Dict[str, Any],
    token_ids: jnp.ndarray,
    timesteps: jnp.ndarray,
    num_heads: int = 8
) -> jnp.ndarray:
    """
    Forward pass of the Diffusion Transformer architecture.

    Args:
        params: Model parameter PyTree.
        token_ids: Discrete integer tokens of shape (B, L), potentially containing `<mask`>.
        timesteps: Continuous diffusion times t in [0, 1] for each sample (B,).
        num_heads: Number of attention heads.

    Returns:
        Unnormalized vocabulary logits of shape (B, L, vocab_size).
    """
    B, L = token_ids.shape
    d_model = params["wte"].shape[1]

    # --- Step 1: Token and Position Embeddings ---
    positions = jnp.arange(0, L)[None, :]  # Shape: (1, L)
    tok_emb = params["wte"][token_ids]     # Shape: (B, L, d_model)
    pos_emb = params["wpe"][positions]     # Shape: (1, L, d_model)
    h = tok_emb + pos_emb                  # Initial hidden states: (B, L, d_model)

    # --- Step 2: Compute Timestep Conditioning Vector ---
    t_emb = sinusoidal_timestep_embedding(timesteps, d_model)  # (B, d_model)
    t_hidden = gelu(jnp.matmul(t_emb, params["time_mlp"]["w1"]) + params["time_mlp"]["b1"])
    time_cond = jnp.matmul(t_hidden, params["time_mlp"]["w2"]) + params["time_mlp"]["b2"]  # (B, d_model)

    # Broadcast conditioning vector across sequence dimension: (B, 1, d_model)
    time_cond = time_cond[:, None, :]

    # --- Step 3: Stacked Transformer Blocks ---
    for layer in params["layers"]:
        # Sub-block 1: Pre-LN + Time injection + Bidirectional Self-Attention + Residual
        norm1 = layer_norm(h, layer["ln1_scale"], layer["ln1_bias"]) + time_cond
        attn_out = bidirectional_attention(
            norm1,
            layer["qkv_w"],
            layer["qkv_b"],
            layer["out_w"],
            layer["out_b"],
            num_heads=num_heads
        )
        h = h + attn_out

        # Sub-block 2: Pre-LN + Time injection + Feed-Forward MLP + Residual
        norm2 = layer_norm(h, layer["ln2_scale"], layer["ln2_bias"]) + time_cond
        mlp_h = gelu(jnp.matmul(norm2, layer["fc1_w"]) + layer["fc1_b"])
        mlp_out = jnp.matmul(mlp_h, layer["fc2_w"]) + layer["fc2_b"]
        h = h + mlp_out

    # --- Step 4: Final LayerNorm & Unembedding Projection ---
    h = layer_norm(h, params["ln_f_scale"], params["ln_f_bias"])
    logits = jnp.matmul(h, params["head_w"]) + params["head_b"]  # (B, L, vocab_size)

    return logits
