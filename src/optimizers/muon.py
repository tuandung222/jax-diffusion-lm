"""
================================================================================
MÔ-ĐUN TỐI ƯU HÓA: MUON OPTIMIZER (MOMENTUM ORTHOGONALIZED BY NEWTON-SCHULZ)
================================================================================
Triển khai thuần JAX (Pure JAX) 100%, không phụ thuộc vào Flax, Haiku hay Optax.
Thuật toán được đề xuất bởi Keller Jordan (2024) và nổi tiếng qua dự án modded-nanogpt.

TẠI SAO LẠI DÙNG MUON?
---------------------
Trong các mô hình Transformer hiện đại (như Llama, GPT, NanoGPT):
1. Các ma trận trọng số 2D (Linear layers, Attention QKV, MLP projections) chứa phần
   lớn tham số của mô hình.
2. Standard SGD hoặc AdamW thường gặp hiện tượng các giá trị suy biến (singular values)
   của ma trận trọng số phát triển không đồng đều; một số hướng gradient có phổ quá lớn,
   dẫn đến mất ổn định (training instability) hoặc hội tụ dưới mức tối ưu.
3. Muon giải quyết bài toán này bằng cách:
   - Tính momentum của gradient như thông thường: M_t = beta * M_{t-1} + (1 - beta) * G_t
   - "Trực giao hóa" (Orthogonalize) ma trận động lượng M_t trước khi cập nhật:
       O_t = M_t * (M_t^T * M_t)^(-1/2)
     sao cho mọi giá trị suy biến (singular values) của bước cập nhật xấp xỉ bằng 1.
   - Nhờ đó, bước cập nhật phân bố đều năng lượng trên toàn bộ các hướng không gian con,
     giúp mô hình học nhanh hơn gấp 2-3 lần so với AdamW thuần túy!

TẠI SAO DÙNG PHÉP LẶP NEWTON-SCHULZ THAY VÌ SVD?
-----------------------------------------------
Tính toán phân rã SVD chính xác O(m * n^2) hoặc nghịch đảo căn bậc hai ma trận
cực kỳ tốn kém và khó song song hóa trên GPU/TPU/Metal. Phép lặp Newton-Schulz
chỉ sử dụng các phép nhân ma trận (GEMM) - vốn là thế mạnh tuyệt đối của phần cứng AI.
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
    Xấp xỉ thừa số trực giao (Polar Decomposition) O = G * (G^T G)^(-1/2)
    thông qua chuỗi đa thức bậc 5 (Quintic Newton-Schulz iteration).

    TOÁN HỌC ĐẰNG SAU PHÉP LẶP:
    --------------------------
    Giả sử ma trận G có dạng phân rã giá trị suy biến G = U * Sigma * V^T.
    Thừa số trực giao gần nhất của G theo chuẩn Frobenius chính là O = U * V^T.
    Để biến đổi Sigma -> I mà không cần tính U, V, ta dùng phép lặp Newton-Schulz:
    
      X_0 = G / (||G||_F + eps)  (chuẩn hóa để các giá trị riêng nằm trong bán kính hội tụ)
      X_{k+1} = a * X_k + b * (X_k X_k^T) * X_k + c * (X_k X_k^T)^2 * X_k

    Các hệ số đa thức quintic tối ưu (được Keller Jordan kiểm chứng thực nghiệm):
      a = 3.4445
      b = -4.7750
      c = 2.0315
    Đa thức f(s) = a*s + b*s^3 + c*s^5 ánh xạ các giá trị suy biến s in [0, 1]
    nhanh chóng tiệm cận về 1.0 chỉ sau 5 bước lặp!

    TỐI ƯU HÓA HÌNH DẠNG (SHAPE AWARENESS):
    --------------------------------------
    Nếu m > n (số hàng nhiều hơn số cột), việc tính X * X^T sẽ tạo ma trận kích thước (m, m)
    rất lớn. Ta chuyển vị G -> G^T để ma trận nhân trong luôn có kích thước min(m, n) x min(m, n),
    tiết kiệm đáng kể bộ nhớ VRAM và thời gian tính toán.

    Args:
        G: Ma trận 2D kích thước (m, n).
        steps: Số vòng lặp Newton-Schulz (thường từ 5 đến 6 là đủ hội tụ).
        eps: Epsilon chống chia cho 0.

    Returns:
        Ma trận trực giao hóa cùng kích thước (m, n).
    """
    assert G.ndim == 2, f"Newton-Schulz chỉ áp dụng cho ma trận 2D, nhận shape: {G.shape}"
    m, n = G.shape
    transposed = False

    # Đảm bảo ma trận luôn có số hàng m <= số cột n để tích X @ X^T có kích thước nhỏ nhất
    if m > n:
        G = G.T
        m, n = n, m
        transposed = True

    # Bước 1: Chuẩn hóa Frobenius đưa phổ giá trị suy biến vào vùng hội tụ (spectral radius < sqrt(3))
    norm = jnp.linalg.norm(G) + eps
    X = G / norm

    # Hệ số đa thức bậc 5
    a = 3.4445
    b = -4.7750
    c = 2.0315

    # Bước 2: Lặp bậc 5 thuần phép nhân ma trận
    for _ in range(steps):
        A = jnp.matmul(X, X.T)       # Kích thước (m, m) - kích thước nhỏ
        AA = jnp.matmul(A, A)         # Kích thước (m, m)
        B = b * A + c * AA            # Kích thước (m, m)
        X = a * X + jnp.matmul(B, X)  # Kích thước (m, n)

    # Chuyển vị ngược lại nếu ban đầu m > n
    if transposed:
        X = X.T

    return X


# ==============================================================================
# QUẢN LÝ TRẠNG THÁI (OPTIMIZER STATE) DẠNG PYTREE
# ==============================================================================
# Trong JAX, mọi hàm đều là "pure function" (hàm thuần khiết không có side-effect).
# Vì vậy, trạng thái của Optimizer không được lưu trữ trong thuộc tính self.state
# như PyTorch, mà được đóng gói thành một cấu trúc cây dữ liệu (PyTree) truyền qua lại.

class LeafMuonState(NamedTuple):
    """Lưu động lượng (momentum buffer) cho từng ma trận 2D tối ưu bằng Muon."""
    momentum: jnp.ndarray


class LeafAdamWState(NamedTuple):
    """Lưu moment bậc 1 (m) và moment bậc 2 (v) cho tham số tối ưu bằng AdamW."""
    m: jnp.ndarray
    v: jnp.ndarray


class MuonOptState(NamedTuple):
    """
    Trạng thái toàn cục của Hybrid Optimizer:
    - step: Bộ đếm số bước huấn luyện hiện tại (scalar int32).
    - leaf_states: Cấu trúc PyTree có hình dạng khớp 100% với tham số mô hình (params),
      chứa LeafMuonState hoặc LeafAdamWState tại mỗi nút lá.
    """
    step: jnp.ndarray
    leaf_states: Any


def default_is_muon_leaf(path: Tuple[Any, ...], p: jnp.ndarray) -> bool:
    """
    Hàm phân loại (Predicate) quyết định tham số nào dùng Muon, tham số nào dùng AdamW:
    
    1. DÙNG MUON:
       - Bắt buộc phải là ma trận 2D (p.ndim == 2).
       - Không thuộc lớp Embedding hay Output Head (vì embedding là bảng tra cứu rời rạc,
         gradient rất thưa (sparse), không phù hợp với giả định trực giao của Muon).
       - Ví dụ: Trọng số QKV Attention, Projection weights, MLP weights.
       
    2. DÙNG ADAMW (Fallback):
       - Tham số 1D: Biases (độ lệch), LayerNorm/RMSNorm scale & shift.
       - Token Embeddings & Positional Embeddings.
       - Language Model Head (ma trận chiếu ra vocabulary).
    """
    # Trích xuất đường dẫn tên biến trong PyTree (ví dụ: ('layers', 0, 'fc1_w'))
    path_str = "/".join(str(getattr(k, "key", k)) for k in path).lower()
    
    if "embed" in path_str or "head" in path_str or "wte" in path_str or "wpe" in path_str:
        return False
        
    return p.ndim == 2


def init_muon_state(params: Any, is_muon_fn=default_is_muon_leaf) -> MuonOptState:
    """
    Khởi tạo trạng thái Optimizer tương ứng với cấu trúc tham số mô hình.

    Args:
        params: PyTree chứa tham số mô hình (nested dict/list của jnp.ndarray).
        is_muon_fn: Hàm xác định tham số lá nào sử dụng Muon.

    Returns:
        MuonOptState chứa bộ đếm step và cây trạng thái leaf_states.
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
    Thực hiện 1 bước cập nhật tối ưu hóa (Optimization Step).
    Hàm này hoàn toàn thuần khiết (pure functional), tương thích tuyệt đối với `jax.jit`.

    CHI TIẾT THUẬT TOÁN:
    -------------------
    1. Với các tham số Muon (2D weights):
       - Cập nhật momentum: M_t = beta * M_{t-1} + (1 - beta) * G_t
       - Tính bước trực giao: O_t = zeropower_via_newtonschulz5(M_t)
       - Hệ số tỷ lệ kích thước: scale = max(1.0, sqrt(m / n))
         (giúp bảo toàn độ lớn tương đối giữa các ma trận có tỷ lệ khung hình khác nhau)
       - Cập nhật: W_{t+1} = W_t * (1 - lr * wd) - lr * scale * O_t

    2. Với các tham số AdamW (1D / Embeddings):
       - Moment bậc 1: m_t = beta1 * m_{t-1} + (1 - beta1) * G_t
       - Moment bậc 2: v_t = beta2 * v_{t-1} + (1 - beta2) * G_t^2
       - Hiệu chỉnh sai số (Bias correction):
           m_hat = m_t / (1 - beta1^t),  v_hat = v_t / (1 - beta2^t)
       - Cập nhật: theta_{t+1} = theta_t * (1 - lr * wd) - lr * (m_hat / (sqrt(v_hat) + eps))

    Returns:
        (new_params, new_state): Bộ tham số mới và trạng thái optimizer mới.
    """
    step = state.step + 1
    lr = jnp.asarray(lr, dtype=jnp.float32)

    def _update_leaf(p: jnp.ndarray, g: jnp.ndarray, s: Any) -> Tuple[jnp.ndarray, Any]:
        if isinstance(s, LeafMuonState):
            # --- Nhánh 1: Cập nhật bằng thuật toán MUON ---
            # 1.1 Tích lũy động lượng gradient
            new_mom = muon_momentum * s.momentum + (1.0 - muon_momentum) * g
            
            # 1.2 Trực giao hóa động lượng qua Newton-Schulz
            ortho_update = zeropower_via_newtonschulz5(new_mom, steps=ns_steps)
            
            # 1.3 Hệ số co giãn bảo toàn hình dạng ma trận
            m, n = p.shape
            scale = jnp.maximum(1.0, jnp.sqrt(float(m) / float(n)))
            
            # 1.4 Áp dụng Weight Decay và bước cập nhật tham số
            decayed_p = p * (1.0 - lr * weight_decay)
            new_p = decayed_p - lr * scale * ortho_update
            
            return new_p, LeafMuonState(momentum=new_mom)

        elif isinstance(s, LeafAdamWState):
            # --- Nhánh 2: Cập nhật bằng thuật toán ADAMW ---
            # 2.1 Tích lũy moment bậc 1 và bậc 2
            new_m = adam_beta1 * s.m + (1.0 - adam_beta1) * g
            new_v = adam_beta2 * s.v + (1.0 - adam_beta2) * (g ** 2)

            # 2.2 Hiệu chỉnh sai số do khởi tạo bằng 0 ở các bước đầu
            bc1 = 1.0 - adam_beta1 ** step
            bc2 = 1.0 - adam_beta2 ** step
            m_hat = new_m / bc1
            v_hat = new_v / bc2

            update = m_hat / (jnp.sqrt(v_hat) + adam_eps)
            
            # 2.3 Weight decay chỉ áp dụng cho ma trận >= 2D (như embedding), bỏ qua bias và norm
            wd = weight_decay if p.ndim >= 2 else 0.0
            decayed_p = p * (1.0 - lr * wd)
            new_p = decayed_p - lr * update

            return new_p, LeafAdamWState(m=new_m, v=new_v)
        else:
            raise TypeError(f"Trạng thái optimizer không hợp lệ: {type(s)}")

    # Trải phẳng PyTree (Flattening) để duyệt song song qua từng cặp (param, grad, state)
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

    # Đóng gói cây (Unflatten) trở lại cấu trúc phân cấp ban đầu
    new_params = jax.tree_util.tree_unflatten(tree_def, new_params_flat)
    new_leaf_states = jax.tree_util.tree_unflatten(state_tree_def, new_leaf_states_flat)

    new_state = MuonOptState(step=step, leaf_states=new_leaf_states)
    return new_params, new_state
