"""
================================================================================
UTILITY MODULE: PURE JAX HELPERS & PYTREE INSPECTION
================================================================================
Provides utility functions for counting parameters, inspecting PyTree structures,
batch preparation, and displaying model parameter summaries.
"""

from typing import Dict, Any, Tuple, List
import jax
import jax.numpy as jnp
from .optimizers.muon import default_is_muon_leaf, LeafMuonState, LeafAdamWState


def count_parameters(params: Any) -> Dict[str, int]:
    """
    Counts total model parameters and categorizes them by optimizer algorithm.

    Returns:
        Dictionary containing:
        - "total": Total parameter count.
        - "muon_params": Parameters updated via Muon (2D weight matrices).
        - "adamw_params": Parameters updated via AdamW (embeddings, biases, norms).
    """
    total = sum(x.size for x in jax.tree_util.tree_leaves(params))
    
    muon_count = 0
    adamw_count = 0

    def _inspect_leaf(path, p):
        nonlocal muon_count, adamw_count
        if default_is_muon_leaf(path, p):
            muon_count += p.size
        else:
            adamw_count += p.size

    jax.tree_util.tree_map_with_path(_inspect_leaf, params)

    return {
        "total": total,
        "muon_params": muon_count,
        "adamw_params": adamw_count
    }


def print_model_summary(params: Any):
    """
    Prints a detailed architectural summary table showing parameter paths,
    tensor shapes, and assigned optimizer algorithms.
    """
    counts = count_parameters(params)
    print("=" * 75)
    print(f"{'PARAMETER NAME (PYTREE PATH)':<42} | {'SHAPE':<16} | {'OPTIMIZER':<8}")
    print("-" * 75)

    def _print_leaf(path, p):
        path_str = "/".join(str(getattr(k, "key", k)) for k in path)
        opt_name = "Muon" if default_is_muon_leaf(path, p) else "AdamW"
        shape_str = str(list(p.shape))
        print(f"{path_str:<42} | {shape_str:<16} | {opt_name:<8}")

    jax.tree_util.tree_map_with_path(_print_leaf, params)
    print("=" * 75)
    print(f"Total Parameters:       {counts['total']:,}")
    print(f"- Muon 2D Matrices:     {counts['muon_params']:,} ({counts['muon_params']/counts['total']*100:.1f}%)")
    print(f"- AdamW Biases/Embeds:  {counts['adamw_params']:,} ({counts['adamw_params']/counts['total']*100:.1f}%)")
    print("=" * 75)


def prepare_dataset(text: str, tokenizer: Any, seq_len: int) -> jnp.ndarray:
    """
    Encodes an input text string into a 2D matrix of non-overlapping chunks
    of fixed sequence length `seq_len`.

    Args:
        text: Raw training text.
        tokenizer: Initialized CharTokenizer instance.
        seq_len: Target sequence length for each chunk.

    Returns:
        2D array of shape (num_samples, seq_len) with dtype jnp.int32.
    """
    tokens = tokenizer.encode(text)
    num_chunks = len(tokens) // seq_len
    if num_chunks == 0:
        raise ValueError(f"Input text is too short for sequence length seq_len={seq_len}!")
        
    usable_tokens = tokens[: num_chunks * seq_len]
    dataset = jnp.array(usable_tokens, dtype=jnp.int32).reshape((num_chunks, seq_len))
    return dataset
