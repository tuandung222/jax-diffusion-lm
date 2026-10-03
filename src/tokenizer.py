"""
================================================================================
TOKENIZER MODULE: CHARACTER-LEVEL TOKENIZER FOR DIFFUSION LM
================================================================================
A self-contained character-level tokenizer with zero third-party dependencies
(no HuggingFace Tokenizers, SentencePiece, or Tiktoken needed).

ROLE OF THE TOKENIZER IN DIFFUSION LANGUAGE MODELING:
---------------------------------------------------
Unlike autoregressive next-token prediction (GPT), Discrete Masked Diffusion
operates with an Absorbing State:

1. The `<mask`> special token serves as the maximum entropy state (Pure Noise).
   At timestep t = 1.0, the sequence is composed entirely of `<mask`> tokens.
2. The neural network learns to recover the original characters at masked locations.
3. `<pad>` is used for batch sequence padding.
4. `<bos>` and `<eos>` denote the beginning and end of text sequences.
5. `<unk>` handles any unseen or out-of-vocabulary characters.

WHY CHARACTER TOKENIZATION FOR EDUCATIONAL STUDY?
------------------------------------------------
- Compact vocabulary size (~50-150 tokens vs. 32,000-128,000 for BPE).
- Dramatically lowers memory footprint and softmax overhead, enabling full training
  and inference directly on laptop CPUs or Apple Silicon M4 without OOM risks.
- Provides a clean, transparent visualization of the unmasking/denoising trajectory!
"""

from typing import List, Dict, Optional, Any, Union


class CharTokenizer:
    """
    Character-level tokenizer supporting special tokens for Masked Diffusion LM.
    """
    
    PAD_TOKEN = "<pad>"
    MASK_TOKEN = "<mask>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, texts: Optional[List[str]] = None):
        """
        Initializes the character vocabulary from sample texts or default ASCII.

        Args:
            texts: List of strings used to build the unique character set.
                   If None, defaults to printable ASCII characters.
        """
        self.special_tokens = [
            self.PAD_TOKEN,
            self.MASK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]
        
        if texts:
            unique_chars = sorted(list(set("".join(texts))))
        else:
            unique_chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
            
        all_tokens = self.special_tokens + [c for c in unique_chars if c not in self.special_tokens]
        
        self.char_to_id: Dict[str, int] = {c: i for i, c in enumerate(all_tokens)}
        self.id_to_char: Dict[int, str] = {i: c for i, c in enumerate(all_tokens)}
        
        # Pre-cache IDs of special tokens for O(1) lookup
        self.pad_id = self.char_to_id[self.PAD_TOKEN]
        self.mask_id = self.char_to_id[self.MASK_TOKEN]
        self.bos_id = self.char_to_id[self.BOS_TOKEN]
        self.eos_id = self.char_to_id[self.EOS_TOKEN]
        self.unk_id = self.char_to_id[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        """Converts an input string into a list of integer token IDs."""
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, token_ids: Any) -> str:
        """
        Decodes a list or array of token IDs back into a human-readable string.
        Accepts List[int], numpy arrays, or JAX DeviceArrays.

        VISUALIZATION NOTE:
        - Remaining `<mask`> tokens are rendered as the block character '█'
          to make the step-by-step unmasking trajectory intuitive and visually clear.
        - Special tokens (<pad>, <bos>, <eos>) are omitted from rendered text.
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        chars = []
        for tid in token_ids:
            tid_int = int(tid)
            char = self.id_to_char.get(tid_int, "")
            if char == self.MASK_TOKEN:
                chars.append("█")  # Visual representation of masked token
            elif char not in self.special_tokens:
                chars.append(char)
        return "".join(chars)
