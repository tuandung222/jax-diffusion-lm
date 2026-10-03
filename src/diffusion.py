"""
================================================================================
MÔ-ĐUN QUY TRÌNH KHUẾCH TÁN: DISCRETE MASKED DIFFUSION PROCESS (MDLM)
================================================================================
Triển khai quy trình khuếch tán rời rạc với trạng thái hấp thụ (Absorbing State Diffusion),
lấy cảm hứng từ các công trình nghiên cứu hiện đại như D3PM, Mask-Predict và MDLM (2024).

SO SÁNH DIFFUSION LIÊN TỤC (ẢNH) VS DIFFUSION RỜI RẠC (NGÔN NGỮ):
---------------------------------------------------------------
1. TRONG XỬ LÝ ẢNH (Continuous Diffusion - DDPM):
   - Không gian điểm ảnh là liên tục (R^D).
   - Quá trình thêm nhiễu (Forward Process) là cộng nhiễu Gauss: x_t = sqrt(alpha_t)*x_0 + sqrt(1 - alpha_t)*eps.
   - Mô hình học cách dự đoán vector nhiễu eps để trừ ngược lại.

2. TRONG MÔ HÌNH NGÔN NGỮ (Discrete Diffusion LM):
   - Ngôn ngữ là các token rời rạc trong từ vựng hữu hạn V.
   - Việc nhúng token vào không gian vector rồi cộng nhiễu Gauss thường gây ra lỗi làm tròn (rounding error)
     khi chiếu ngược về từ vựng, dẫn đến câu văn vô nghĩa.
   - Giải pháp ưu việt nhất hiện nay là "Masked Diffusion":
     * Trạng thái hấp thụ (Absorbing State): Token `<mask`>.
     * Quá trình khuếch tán tiến (Forward Noising): Ở thời điểm t in [0, 1], mỗi từ có xác suất p=t
       bị biến thành token `<mask`>.
     * Tại t=0: Câu gốc hoàn chỉnh x_0 (sạch 100%).
     * Tại t=1: Câu chỉ toàn là `<mask`> (nhiễu 100%).
     * Quá trình khuếch tán ngược (Reverse Sampling): Bắt đầu từ câu toàn `<mask`>, mô hình nơ-ron
       dự đoán phân phối các từ gốc và từng bước gỡ mặt nạ (unmask) để tạo thành văn bản hoàn chỉnh!

LỢI THẾ CỦA MASKED DIFFUSION SO VỚI AUTOREGRESSIVE (GPT):
--------------------------------------------------------
- Khả năng điền từ hai chiều (Infilling / Editing): Có thể giữ cố định một số từ và cho mô hình
  khuếch tán sinh các từ còn thiếu ở bất kỳ vị trí nào trong câu.
- Khả năng kiểm soát độ dài và cấu trúc câu linh hoạt.
- Tốc độ sinh có thể tùy chỉnh: Có thể sinh câu trong 10-30 bước thay vì phải chạy L bước tuần tự như GPT.
"""

from typing import Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
from .model import forward_transformer


def q_sample(
    rng: jax.random.PRNGKey,
    x_0: jnp.ndarray,
    t: jnp.ndarray,
    mask_token_id: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quá trình khuếch tán tiến q(x_t | x_0, t): Thêm nhiễu bằng cách che mặt nạ.

    NGUYÊN LÝ:
    ----------
    Với mỗi mẫu trong batch tại thời điểm t in [0, 1], ta tạo một biến ngẫu nhiên
    Bernoulli độc lập cho từng vị trí token với xác suất thành công p = t.
    Nếu trúng, token được thay thế bằng `mask_token_id`.

    Args:
        rng: Khóa ngẫu nhiên PRNGKey.
        x_0: Mảng token sạch ban đầu, shape (batch_size, seq_len).
        t: Mảng 1D chứa bước thời gian khuếch tán của từng câu trong batch, shape (batch_size,).
        mask_token_id: ID số nguyên của token `<mask`>.

    Returns:
        x_t: Mảng token bị làm nhiễu, shape (batch_size, seq_len).
        mask_indices: Mảng boolean (batch_size, seq_len), True tại các vị trí bị che mặt nạ.
    """
    # Mở rộng kích thước t để broadcast dọc theo chiều dài chuỗi: (B, 1)
    p_mask = t[:, None]
    
    # Tung đồng xu Bernoulli độc lập cho từng token trong batch
    mask_indices = jax.random.bernoulli(rng, p=p_mask, shape=x_0.shape)
    
    # Thay thế các vị trí True bằng mask_token_id
    x_t = jnp.where(mask_indices, mask_token_id, x_0)
    
    return x_t, mask_indices


def compute_loss(
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    x_0: jnp.ndarray,
    mask_token_id: int,
    pad_token_id: int = 0,
    num_heads: int = 8,
    min_t: float = 1e-4
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Tính hàm mất mát Cross-Entropy dựa trên biến phân (Variational Lower Bound - ELBO).

    BẢN CHẤT TOÁN HỌC:
    ------------------
    Trong Discrete Masked Diffusion, hàm mục tiêu cực đại hóa cận dưới của hợp lý log (ELBO)
    được rút gọn thành bài toán phân loại đa lớp (Cross-Entropy):
    
      L(theta) = E_{t ~ U(0, 1), x_t ~ q(x_t|x_0)} [ -log p_theta(x_0 | x_t, t) ]

    QUAN TRỌNG:
    - Ta CHỈ tính loss trên các vị trí thực sự bị che mặt nạ (`mask_indices == True`).
    - Các vị trí không bị che (`x_t[i] == x_0[i]`) đã lộ rõ thông tin cho mô hình, nếu tính
      loss ở đây mô hình sẽ học thói quen sao chép tầm thường (trivial copying).
    - Đồng thời, ta loại trừ các token `<pad>` để không phạt mô hình ở phần đệm câu.

    Args:
        params: PyTree tham số mô hình.
        rng: Khóa ngẫu nhiên PRNGKey.
        x_0: Mảng token nhãn gốc, shape (B, L).
        mask_token_id: ID của token `<mask`>.
        pad_token_id: ID của token `<pad>` (không tính loss).
        num_heads: Số đầu chú ý của Transformer.
        min_t: Ngưỡng thời gian nhỏ nhất (tránh t=0 tuyệt đối gây suy biến).

    Returns:
        loss: Giá trị mất mát vô hướng dùng để lan truyền ngược (Backpropagation).
        metrics: Từ điển chứa các số đo theo dõi (loss, masked_acc, masked_ratio).
    """
    B, L = x_0.shape
    rng_t, rng_noise = jax.random.split(rng)

    # 1. Lấy mẫu thời gian ngẫu nhiên t đồng đều trong [min_t, 1.0] cho từng câu trong batch
    t = jax.random.uniform(rng_t, shape=(B,), minval=min_t, maxval=1.0)

    # 2. Tạo chuỗi bị làm nhiễu x_t
    x_t, mask_indices = q_sample(rng_noise, x_0, t, mask_token_id)

    # 3. Lan truyền tiến qua Transformer để dự đoán phân phối từ gốc
    logits = forward_transformer(params, x_t, t, num_heads=num_heads)  # (B, L, vocab_size)

    # 4. Tính Cross-Entropy Loss
    # log_softmax chuyển đổi logits thành log-xác suất ổn định số học
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, L, vocab_size)
    
    # Trích xuất log-xác suất tương ứng với token gốc x_0 tại mỗi vị trí
    target_log_probs = jnp.take_along_axis(log_probs, x_0[..., None], axis=-1).squeeze(-1)  # (B, L)

    # Mặt nạ hợp lệ: Chỉ tính trên các vị trí bị MASK và khác PAD
    valid_mask = mask_indices & (x_0 != pad_token_id)
    num_valid = jnp.maximum(1.0, jnp.sum(valid_mask))
    
    # Trung bình cộng Cross-Entropy trên toàn bộ token bị che hợp lệ
    loss = -jnp.sum(target_log_probs * valid_mask) / num_valid

    # 5. Đánh giá độ chính xác (Accuracy) trên các token bị che để tiện giám sát quá trình học
    pred_tokens = jnp.argmax(logits, axis=-1)
    correct = (pred_tokens == x_0) & valid_mask
    acc = jnp.sum(correct) / num_valid

    metrics = {
        "loss": loss,
        "masked_acc": acc,
        "masked_ratio": jnp.mean(mask_indices)
    }
    return loss, metrics


def sample_tokens(
    params: Dict[str, Any],
    rng: jax.random.PRNGKey,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    mask_token_id: int,
    num_heads: int = 8,
    temperature: float = 1.0
) -> jnp.ndarray:
    """
    Quy trình khuếch tán ngược (Reverse Ancestral Sampling) để sinh văn bản mới.

    NGUYÊN LÝ SINH CÂU TỪ NHIỄU THUẦN TÚY:
    --------------------------------------
    1. Khởi đầu tại t=1.0: Chuỗi là một mảng toàn bộ là token `<mask`> (nhiễu 100%).
    2. Chia khoảng thời gian [1.0 -> 0.0] thành `num_steps` bước nhảy: dt = 1.0 / num_steps.
    3. Tại mỗi bước lặp từ t xuống t_next:
       - Đưa chuỗi hiện tại x_t vào Transformer để dự đoán toàn bộ câu gốc x_0.
       - Lấy mẫu ứng viên từ phân phối Categorical(logits / temperature).
       - Tính xác suất mở mặt nạ (Unmask Probability):
           p_unmask = (t - t_next) / t
         (Đây là công thức giải tích chuẩn đảm bảo phân phối biên đúng theo thời gian).
       - Đối với những vị trí đang là `<mask`>, gieo xác suất p_unmask:
         * Nếu mở: gán giá trị dự đoán mới.
         * Nếu chưa: giữ nguyên `<mask`> để các bước sau tiếp tục hoàn thiện dựa trên ngữ cảnh mới!
    4. Tại bước cuối cùng (t -> 0): Mở toàn bộ các mask còn sót lại để hoàn thành văn bản.

    Args:
        params: Trọng số mô hình đã huấn luyện.
        rng: Khóa ngẫu nhiên PRNGKey.
        batch_size: Số lượng câu cần sinh song song.
        seq_len: Độ dài chuỗi ký tự muốn sinh.
        num_steps: Số bước khuếch tán ngược (ví dụ: 16, 32, 64 bước).
        mask_token_id: ID của token `<mask`>.
        num_heads: Số đầu chú ý của Transformer.
        temperature: Nhiệt độ lấy mẫu (temperature > 1: phong phú hơn; temperature < 1: chính xác hơn).

    Returns:
        Mảng token ID hoàn chỉnh sau khi khử nhiễu, shape (batch_size, seq_len).
    """
    # Khởi tạo trạng thái ban đầu: 100% là token [MASK]
    x = jnp.full((batch_size, seq_len), mask_token_id, dtype=jnp.int32)
    dt = 1.0 / float(num_steps)

    for step in range(num_steps):
        # Tính bước thời gian hiện tại t và bước kế tiếp t_next
        t_val = 1.0 - step * dt
        t_next_val = jnp.maximum(0.0, t_val - dt)
        t_arr = jnp.full((batch_size,), t_val, dtype=jnp.float32)

        rng, rng_step, rng_unmask = jax.random.split(rng, 3)

        # 1. Dự đoán phân phối các từ gốc x_0 dựa trên ngữ cảnh hai chiều của các từ đã lộ diện
        logits = forward_transformer(params, x, t_arr, num_heads=num_heads)
        
        # 2. Điều chỉnh nhiệt độ và lấy mẫu ứng viên ngẫu nhiên (Categorical Sampling)
        scaled_logits = logits / jnp.maximum(1e-5, temperature)
        x_0_pred = jax.random.categorical(rng_step, scaled_logits, axis=-1)

        # 3. Xác định vị trí nào sẽ được gỡ bỏ mặt nạ trong bước này
        is_masked = (x == mask_token_id)
        if step == num_steps - 1:
            # Bước cuối cùng: mở toàn bộ mọi vị trí còn lại
            unmask_prob = 1.0
        else:
            # Xác suất gỡ mặt nạ theo tỷ lệ bước nhảy thời gian
            unmask_prob = (t_val - t_next_val) / t_val

        unmask_flags = jax.random.bernoulli(rng_unmask, p=unmask_prob, shape=(batch_size, seq_len))
        should_unmask = is_masked & unmask_flags

        # 4. Cập nhật chuỗi: thay thế vị trí được mở bằng từ ứng viên dự đoán
        x = jnp.where(should_unmask, x_0_pred, x)

    return x
