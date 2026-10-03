"""
================================================================================
MÔ-ĐUN KIẾN TRÚC MÔ HÌNH: PURE JAX BIDIRECTIONAL DIFFUSION TRANSFORMER
================================================================================
Triển khai mô hình Transformer hai chiều (Bidirectional Transformer) thuần JAX 100%.
Không sử dụng bất kỳ framework cấp cao nào như Flax, Haiku, Keras hay PyTorch.

ĐẶC TRƯNG CỐT LÕI CỦA MẠNG DIFFUSION LANGUAGE MODEL:
--------------------------------------------------
1. CHÚ Ý HAI CHIỀU (BIDIRECTIONAL ATTENTION):
   - Trong mô hình sinh tuần tự (Autoregressive - GPT), ma trận chú ý phải dùng mặt
     nạ tam giác dưới (Causal Mask) để ngăn token hiện tại nhìn thấy tương lai.
   - Ngược lại, trong Diffusion LM, chuỗi đầu vào x_t đã bị che ngẫu nhiên ở nhiều vị trí.
     Để dự đoán chính xác một từ bị che ở giữa câu, mô hình PHẢI nhìn thấy cả ngữ cảnh
     bên trái (tiền tố) và bên phải (hậu tố).
   - Do đó, ta sử dụng cơ chế Self-Attention hai chiều đầy đủ (Full Dense Attention),
     tương tự như BERT hay RoBERTa.

2. ĐIỀU KIỆN HÓA BƯỚC KHUẾCH TÁN (TIMESTEP CONDITIONING):
   - Mô hình khuếch tán cần biết mức độ nhiễu hiện tại t in [0, 1] để điều chỉnh dự đoán:
     * Khi t gần 1.0 (chuỗi gần như bị che toàn bộ): Mô hình dự đoán dựa trên xác suất tiên nghiệm.
     * Khi t gần 0.0 (chỉ còn vài token bị che): Mô hình dựa vào ngữ cảnh cục bộ để điền từ chính xác.
   - Bước thời gian t được mã hóa thành vector thông qua Sinusoidal Timestep Embeddings
     kết hợp với mạng Time-MLP, sau đó được hòa trộn vào từng khối Transformer.

3. TRIẾT LÝ HÀM THUẦN KHIẾT (FUNCTIONAL PURITY TRONG JAX):
   - Không dùng `class Model(nn.Module)`.
   - Tham số mô hình được lưu trong cấu trúc từ điển PyTree đơn thuần (`params = {...}`).
   - Hai hàm chính: `init_transformer_params` (khởi tạo) và `forward_transformer` (lan truyền tiến).
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
    Mã hóa bước khuếch tán liên tục t in [0, 1] thành vector nhiều tần số (Sinusoidal Embedding).
    
    Ý TƯỞNG TOÁN HỌC:
    -----------------
    Tương tự như Positional Encoding trong bài báo "Attention Is All You Need",
    ta chiếu đại lượng vô hướng t lên các sóng hình sin và cosin với các dải tần
    số hình học giảm dần từ cao xuống thấp:
    
      omega_k = exp(-log(max_period) * k / (dim // 2))
      emb = [cos(1000 * t * omega), sin(1000 * t * omega)]

    Nhờ đó, mạng nơ-ron có thể phân biệt cực kỳ nhạy bén giữa các mức độ nhiễu
    t khác nhau (ví dụ: t=0.01 rất khác t=0.02).

    Args:
        timesteps: Mảng 1D chứa giá trị thời gian t của batch, shape (batch_size,).
        dim: Số chiều vector nhúng mong muốn (thường bằng d_model).
        max_period: Chu kỳ tối đa điều khiển dải tần số thấp nhất.

    Returns:
        Mảng 2D chứa vector nhúng thời gian, shape (batch_size, dim).
    """
    half = dim // 2
    # Dải tần số góc giảm dần theo cấp số nhân
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    # Nhân với 1000 để kéo giãn thang đo liên tục t in [0, 1] thành dải phổ rộng
    args = timesteps[:, None] * freqs[None, :] * 1000.0  # (batch_size, half)
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)  # (batch_size, 2 * half)
    
    # Trường hợp số chiều lẻ, bù thêm 1 cột 0 ở cuối
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
    Chuẩn hóa theo tầng (Layer Normalization) trên chiều đặc trưng cuối cùng.
    
    Công thức:
      y = ((x - mean) / sqrt(var + eps)) * scale + bias
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / jnp.sqrt(var + eps)
    return normed * scale + bias


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Hàm kích hoạt GELU (Gaussian Error Linear Unit) xấp xỉ nhanh.
    Được sử dụng phổ biến trong BERT, GPT-2, RoBERTa.
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
    Khởi tạo toàn bộ ma trận trọng số cho Transformer Diffusion Backbone.
    
    PHÂN BỐ TRỌNG SỐ KHỞI TẠO:
    - Token Embeddings & Positional Embeddings: Phân phối chuẩn N(0, 0.02^2).
    - Linear Layers (Attention, MLP): Chuẩn hóa Xavier/He scaled N(0, 1 / sqrt(fan_in)).
    - Biases: Khởi tạo hoàn toàn bằng 0.
    - LayerNorm scales: Khởi tạo bằng 1.

    Args:
        rng: JAX PRNGKey (Khóa ngẫu nhiên bất biến của JAX).
        vocab_size: Tổng số lượng token trong từ vựng.
        d_model: Số chiều vector ẩn (Hidden dimension).
        num_heads: Số lượng đầu chú ý (Attention heads).
        num_layers: Số lượng khối Transformer xếp chồng.
        d_ff: Số chiều ẩn của lớp mạng Feed-Forward (MLP dimension, thường = 4 * d_model).
        max_len: Chiều dài tối đa của chuỗi văn bản.

    Returns:
        PyTree Dictionary chứa toàn bộ trọng số của mô hình.
    """
    params = {}
    
    # 1. Bảng nhúng Token và Vị trí
    rng, k_tok, k_pos = jax.random.split(rng, 3)
    params["wte"] = jax.random.normal(k_tok, (vocab_size, d_model)) * 0.02
    params["wpe"] = jax.random.normal(k_pos, (max_len, d_model)) * 0.02

    # 2. Mạng MLP điều kiện hóa bước khuếch tán (Time MLP)
    # Ánh xạ từ Sinusoidal Embedding (d_model) -> 2*d_model -> d_model
    rng, k_t1, k_t2 = jax.random.split(rng, 3)
    params["time_mlp"] = {
        "w1": jax.random.normal(k_t1, (d_model, d_model * 2)) * 0.02,
        "b1": jnp.zeros(d_model * 2),
        "w2": jax.random.normal(k_t2, (d_model * 2, d_model)) * 0.02,
        "b2": jnp.zeros(d_model),
    }

    # 3. Các khối Transformer Blocks
    layers = []
    for _ in range(num_layers):
        rng, k_qkv, k_out, k_fc1, k_fc2 = jax.random.split(rng, 5)
        layer_params = {
            # Chuẩn hóa tiền Attention (Pre-LN)
            "ln1_scale": jnp.ones(d_model),
            "ln1_bias": jnp.zeros(d_model),
            # Chiếu Query, Key, Value gộp chung trong 1 ma trận (d_model, 3 * d_model)
            "qkv_w": jax.random.normal(k_qkv, (d_model, 3 * d_model)) * (1.0 / math.sqrt(d_model)),
            "qkv_b": jnp.zeros(3 * d_model),
            # Chiếu đầu ra của Multi-Head Attention
            "out_w": jax.random.normal(k_out, (d_model, d_model)) * (1.0 / math.sqrt(d_model)),
            "out_b": jnp.zeros(d_model),
            # Chuẩn hóa tiền MLP (Pre-LN)
            "ln2_scale": jnp.ones(d_model),
            "ln2_bias": jnp.zeros(d_model),
            # Lớp mạng nơ-ron truyền thẳng (MLP/FFN)
            "fc1_w": jax.random.normal(k_fc1, (d_model, d_ff)) * (1.0 / math.sqrt(d_model)),
            "fc1_b": jnp.zeros(d_ff),
            "fc2_w": jax.random.normal(k_fc2, (d_ff, d_model)) * (1.0 / math.sqrt(d_ff)),
            "fc2_b": jnp.zeros(d_model),
        }
        layers.append(layer_params)
    params["layers"] = layers

    # 4. Chuẩn hóa cuối cùng và Đầu dự đoán từ vựng (Unembedding Head)
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
    Cơ chế Self-Attention hai chiều đa đầu (Multi-Head Bidirectional Self-Attention).

    DÒNG CHẢY DỮ LIỆU VÀ KÍCH THƯỚC (TENSOR SHAPES):
    -----------------------------------------------
    - x: (B, L, D) trong đó B=Batch, L=SeqLen, D=HiddenDim.
    - Chiếu tuyến tính QKV: (B, L, D) @ (D, 3*D) -> (B, L, 3*D).
    - Tách thành 3 ma trận Q, K, V mỗi ma trận: (B, L, D).
    - Tách theo số đầu chú ý: (B, num_heads, L, head_dim) với head_dim = D // num_heads.
    - Điểm tương đồng: Scores = (Q @ K^T) / sqrt(head_dim) -> (B, num_heads, L, L).
    - Trọng số chú ý: Softmax(Scores) -> (B, num_heads, L, L).
    - Ngữ cảnh tổng hợp: Context = Trọng số @ V -> (B, num_heads, L, head_dim).
    - Gom các đầu chú ý và chiếu đầu ra: Context -> (B, L, D) @ (D, D) -> (B, L, D).
    """
    B, L, D = x.shape
    head_dim = D // num_heads

    # 1. Tính toán song song Q, K, V thông qua một phép nhân ma trận duy nhất
    qkv = jnp.matmul(x, qkv_w) + qkv_b                     # (B, L, 3 * D)
    q, k, v = jnp.split(qkv, 3, axis=-1)                  # 3 mảng (B, L, D)

    # 2. Phân bổ về từng đầu chú ý (Multi-Head Reshape & Transpose)
    q = jnp.transpose(jnp.reshape(q, (B, L, num_heads, head_dim)), (0, 2, 1, 3))  # (B, H, L, d_k)
    k = jnp.transpose(jnp.reshape(k, (B, L, num_heads, head_dim)), (0, 2, 1, 3))  # (B, H, L, d_k)
    v = jnp.transpose(jnp.reshape(v, (B, L, num_heads, head_dim)), (0, 2, 1, 3))  # (B, H, L, d_k)

    # 3. Scaled Dot-Product Attention (Không áp dụng Causal Mask để nhìn thấy hai chiều)
    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale  # (B, H, L, L)
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # 4. Nhân với vector Value để thu được biểu diễn ngữ cảnh
    context = jnp.matmul(attn_weights, v)  # (B, H, L, d_k)
    context = jnp.transpose(context, (0, 2, 1, 3))
    context = jnp.reshape(context, (B, L, D))

    # 5. Chiếu qua ma trận đầu ra
    return jnp.matmul(context, out_w) + out_b


def forward_transformer(
    params: Dict[str, Any],
    token_ids: jnp.ndarray,
    timesteps: jnp.ndarray,
    num_heads: int = 8
) -> jnp.ndarray:
    """
    Lan truyền tiến (Forward Pass) của toàn bộ kiến trúc Diffusion Transformer.

    Args:
        params: PyTree chứa toàn bộ trọng số mô hình.
        token_ids: Mảng số nguyên chứa các token đầu vào (B, L), có thể chứa token `<mask`>.
        timesteps: Mảng 1D chứa bước thời gian khuếch tán t in [0, 1] của từng mẫu (B,).
        num_heads: Số đầu chú ý trong Multi-Head Attention.

    Returns:
        Logits chưa qua Softmax có shape (B, L, vocab_size).
    """
    B, L = token_ids.shape
    d_model = params["wte"].shape[1]

    # --- Bước 1: Nhúng Token và Tọa độ Vị trí ---
    positions = jnp.arange(0, L)[None, :]  # (1, L)
    tok_emb = params["wte"][token_ids]     # (B, L, d_model)
    pos_emb = params["wpe"][positions]     # (1, L, d_model)
    h = tok_emb + pos_emb                  # Trạng thái ẩn ban đầu (B, L, d_model)

    # --- Bước 2: Tạo Vector Điều kiện hóa Thời gian (Time Conditioning) ---
    t_emb = sinusoidal_timestep_embedding(timesteps, d_model)  # (B, d_model)
    t_hidden = gelu(jnp.matmul(t_emb, params["time_mlp"]["w1"]) + params["time_mlp"]["b1"])
    time_cond = jnp.matmul(t_hidden, params["time_mlp"]["w2"]) + params["time_mlp"]["b2"]  # (B, d_model)

    # Mở rộng chiều sequence để broadcast cộng vào mọi vị trí trong câu: (B, 1, d_model)
    time_cond = time_cond[:, None, :]

    # --- Bước 3: Đi qua các khối Transformer Blocks ---
    for layer in params["layers"]:
        # Khối con 1: Pre-LayerNorm + Điều kiện hóa thời gian + Self-Attention + Kết nối tắt (Residual)
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

        # Khối con 2: Pre-LayerNorm + Điều kiện hóa thời gian + Feed-Forward MLP + Kết nối tắt (Residual)
        norm2 = layer_norm(h, layer["ln2_scale"], layer["ln2_bias"]) + time_cond
        mlp_h = gelu(jnp.matmul(norm2, layer["fc1_w"]) + layer["fc1_b"])
        mlp_out = jnp.matmul(mlp_h, layer["fc2_w"]) + layer["fc2_b"]
        h = h + mlp_out

    # --- Bước 4: Chuẩn hóa cuối cùng và Chiếu ra Logits Từ vựng ---
    h = layer_norm(h, params["ln_f_scale"], params["ln_f_bias"])
    logits = jnp.matmul(h, params["head_w"]) + params["head_b"]  # (B, L, vocab_size)

    return logits
