"""
Pure JAX Diffusion Language Model Package.
"""
from .tokenizer import CharTokenizer
from .model import init_transformer_params, forward_transformer, bidirectional_attention
from .diffusion import q_sample, compute_loss, sample_tokens
from .block_diffusion import (
    make_block_causal_mask,
    block_q_sample,
    compute_block_loss,
    sample_block_diffusion
)
from .optimizers import init_muon_state, muon_step, cosine_warmup_schedule
from .utils import count_parameters, print_model_summary, prepare_dataset

__all__ = [
    "CharTokenizer",
    "init_transformer_params",
    "forward_transformer",
    "bidirectional_attention",
    "q_sample",
    "compute_loss",
    "sample_tokens",
    "make_block_causal_mask",
    "block_q_sample",
    "compute_block_loss",
    "sample_block_diffusion",
    "init_muon_state",
    "muon_step",
    "cosine_warmup_schedule",
    "count_parameters",
    "print_model_summary",
    "prepare_dataset",
]
