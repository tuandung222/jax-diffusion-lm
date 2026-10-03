"""
================================================================================
MÔ-ĐUN LẬP LỊCH TỐC ĐỘ HỌC: LEARNING RATE SCHEDULER
================================================================================
Triển khai thuần JAX (Pure JAX) 100%, tính toán dạng hàm thuần túy (pure function)
và hoàn toàn tương thích với trình biên dịch XLA (`jax.jit`).

TẠI SAO PHẢI CÓ WARMUP + COSINE DECAY TRONG TRANSFORMER & DIFFUSION?
-------------------------------------------------------------------
1. GIAI ĐOẠN WARMUP (Khởi động tuyến tính):
   - Khi vừa khởi tạo, các ma trận trọng số trong mô hình Transformer còn ở trạng
     thái ngẫu nhiên. Lúc này gradient ban đầu có độ biến thiên rất lớn (noisy gradients).
   - Nếu áp dụng ngay tốc độ học cực đại (base_lr), các bước cập nhật ban đầu có thể
     làm chệch hướng trọng số đến những vùng mất ổn định vĩnh viễn.
   - Warmup bắt đầu từ 0 và tăng dần tuyến tính lên base_lr trong `warmup_steps` bước đầu,
     giúp mô hình dần ổn định phương sai trước khi tăng tốc huấn luyện.

2. GIAI ĐOẠN COSINE DECAY (Suy giảm theo hàm Cosine):
   - Sau khi hoàn thành warmup, mô hình đã đi vào thung lũng nghiệm tốt.
   - Việc hạ dần tốc độ học theo quỹ đạo nửa chu kỳ hàm cosin giúp mô hình tinh chỉnh
     sâu vào đáy cực tiểu cục bộ (local minimum), tránh bị dao động qua lại quanh nghiệm.
   - Hàm cosin tạo ra sự chuyển tiếp mượt mà ở cả đầu và cuối chu kỳ (đạo hàm bậc nhất
     bằng 0 tại đỉnh và đáy), vượt trội so với suy giảm tuyến tính (linear decay).

TẠI SAO DÙNG `jnp.where` THAY VÌ `if/else` TRUYỀN THỐNG TRONG PYTHON?
--------------------------------------------------------------------
Trình biên dịch XLA của JAX hoạt động bằng cách "dò vết" (trace) các toán tử mảng.
Lệnh rẽ nhánh `if step < warmup_steps:` của Python dựa trên giá trị cụ thể tại thời
điểm chạy, sẽ thất bại khi `step` là một JAX Dynamic Array bên trong hàm `@jax.jit`.
Hàm `jnp.where` là toán tử chọn lọc không rẽ nhánh (branchless selection), được biên
dịch trực tiếp thành lệnh phần cứng tăng tốc trên GPU/Metal.
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
    Tính toán Learning Rate tại bước `step` theo công thức Linear Warmup + Cosine Decay.

    CÔNG THỨC TOÁN HỌC:
    -------------------
    1. Nếu step < warmup_steps:
         lr(step) = base_lr * (step / warmup_steps)

    2. Nếu step >= warmup_steps:
         progress = (step - warmup_steps) / (total_steps - warmup_steps)
         lr(step) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))

    Args:
        step: Bước huấn luyện hiện tại (scalar int hoặc float jnp.ndarray).
        base_lr: Tốc độ học cực đại (Peak Learning Rate sau khi kết thúc warmup).
        warmup_steps: Số bước khởi động tuyến tính (ví dụ: 10% tổng số bước).
        total_steps: Tổng số bước huấn luyện dự kiến.
        min_lr: Tốc độ học tối thiểu ở cuối quá trình huấn luyện (tránh triệt tiêu về 0).

    Returns:
        Learning rate tương ứng với bước hiện tại dưới dạng scalar `jnp.ndarray`.
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    warmup_steps = float(warmup_steps)
    total_steps = float(total_steps)

    # 1. Quỹ đạo khởi động tuyến tính: tăng dần từ 0 -> base_lr
    # Dùng jnp.maximum(1.0, warmup_steps) để phòng ngừa chia cho 0 nếu warmup_steps = 0
    warmup_lr = base_lr * (step / jnp.maximum(1.0, warmup_steps))

    # 2. Quỹ đạo suy giảm cosin: giảm dần từ base_lr -> min_lr
    decay_steps = jnp.maximum(1.0, total_steps - warmup_steps)
    # jnp.clip đảm bảo progress không bao giờ vượt ra ngoài khoảng [0.0, 1.0]
    progress = jnp.clip((step - warmup_steps) / decay_steps, 0.0, 1.0)
    cosine_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + jnp.cos(jnp.pi * progress))

    # 3. Lựa chọn không rẽ nhánh (Branchless Select) cho XLA JIT
    return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)
