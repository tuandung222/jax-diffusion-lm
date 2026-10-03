"""
================================================================================
MÔ-ĐUN TIỆN ÍCH: PURE JAX UTILS & PYTREE HELPERS
================================================================================
Cung cấp các công cụ hỗ trợ cho việc đếm tham số, phân tích cấu trúc PyTree,
tạo lô huấn luyện (Batching), và in tóm tắt mô hình.
"""

from typing import Dict, Any, Tuple, List
import jax
import jax.numpy as jnp
from .optimizers.muon import default_is_muon_leaf, LeafMuonState, LeafAdamWState


def count_parameters(params: Any) -> Dict[str, int]:
    """
    Đếm số lượng tham số trong mô hình và phân loại theo thuật toán tối ưu hóa.

    Returns:
        Từ điển chứa:
        - "total": Tổng số tham số.
        - "muon_params": Số tham số tối ưu bằng thuật toán Muon (các ma trận 2D).
        - "adamw_params": Số tham số tối ưu bằng AdamW (embeddings, biases, normalizations).
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
    In bảng tóm tắt chi tiết cấu trúc mô hình, kích thước từng tầng và phân bổ Optimizer.
    """
    counts = count_parameters(params)
    print("=" * 75)
    print(f"{'TÊN THAM SỐ (PYTREE PATH)':<42} | {'KÍCH THƯỚC':<16} | {'OPTIMIZER':<8}")
    print("-" * 75)

    def _print_leaf(path, p):
        path_str = "/".join(str(getattr(k, "key", k)) for k in path)
        opt_name = "Muon" if default_is_muon_leaf(path, p) else "AdamW"
        shape_str = str(list(p.shape))
        print(f"{path_str:<42} | {shape_str:<16} | {opt_name:<8}")

    jax.tree_util.tree_map_with_path(_print_leaf, params)
    print("=" * 75)
    print(f"Tổng số tham số:        {counts['total']:,}")
    print(f"- Ma trận tối ưu Muon:  {counts['muon_params']:,} ({counts['muon_params']/counts['total']*100:.1f}%)")
    print(f"- Tham số tối ưu AdamW: {counts['adamw_params']:,} ({counts['adamw_params']/counts['total']*100:.1f}%)")
    print("=" * 75)


def prepare_dataset(text: str, tokenizer: Any, seq_len: int) -> jnp.ndarray:
    """
    Mã hóa chuỗi văn bản dài thành một ma trận các đoạn có độ dài cố định seq_len.

    Args:
        text: Toàn bộ văn bản huấn luyện.
        tokenizer: Đối tượng CharTokenizer.
        seq_len: Độ dài mỗi đoạn văn bản.

    Returns:
        Mảng 2D (num_samples, seq_len) kiểu jnp.int32.
    """
    tokens = tokenizer.encode(text)
    num_chunks = len(tokens) // seq_len
    if num_chunks == 0:
        raise ValueError(f"Văn bản quá ngắn so với độ dài seq_len={seq_len}!")
        
    usable_tokens = tokens[: num_chunks * seq_len]
    dataset = jnp.array(usable_tokens, dtype=jnp.int32).reshape((num_chunks, seq_len))
    return dataset
