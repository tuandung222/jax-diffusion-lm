"""
================================================================================
SCRIPT SINH VĂN BẢN TRỰC QUAN: STEP-BY-STEP DIFFUSION VISUALIZATION
================================================================================
Trình diễn trực quan từng bước unmasking (khử nhiễu) của mô hình Discrete Diffusion LM.
Giúp người học quan sát trực tiếp cách một chuỗi văn bản mạch lạc dần dần hình thành
từ một chuỗi nhiễu mặt nạ đen đặc (████████).
"""

from typing import List, Dict, Any
import jax
import jax.numpy as jnp

from src.tokenizer import CharTokenizer
from src.model import init_transformer_params, forward_transformer
from src.diffusion import sample_tokens


def visualize_sampling_trajectory(
    params: Dict[str, Any],
    tokenizer: CharTokenizer,
    rng: jax.random.PRNGKey,
    seq_len: int = 64,
    num_steps: int = 20,
    temperature: float = 0.8,
    num_heads: int = 4
):
    """
    Chạy quá trình Reverse Diffusion và in ra trạng thái chuỗi ký tự tại từng bước lặp.
    """
    print("=" * 75)
    print(" QUAN SÁT TIẾN TRÌNH KHỬ NHIỄU KHUẾCH TÁN (STEP-BY-STEP DENOISING)")
    print(f" Số bước khuếch tán ngược: {num_steps} | Độ dài: {seq_len} ký tự | Nhiệt độ: {temperature}")
    print("=" * 75)

    # Bắt đầu tại t=1.0: Mọi vị trí đều là [MASK] (hiển thị bằng ký tự █)
    x = jnp.full((1, seq_len), tokenizer.mask_id, dtype=jnp.int32)
    dt = 1.0 / float(num_steps)

    print(f"Bước [00/{num_steps:02d}] (t=1.00 - Nhiễu 100%):")
    print(f"  {tokenizer.decode(list(x[0]))}\n")

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t_next_val = jnp.maximum(0.0, t_val - dt)
        t_arr = jnp.array([t_val], dtype=jnp.float32)

        rng, rng_step, rng_unmask = jax.random.split(rng, 3)

        # 1. Dự đoán phân phối sạch x_0
        logits = forward_transformer(params, x, t_arr, num_heads=num_heads)
        scaled_logits = logits / jnp.maximum(1e-5, temperature)
        x_0_pred = jax.random.categorical(rng_step, scaled_logits, axis=-1)

        # 2. Tính xác suất mở mặt nạ
        is_masked = (x == tokenizer.mask_id)
        if step == num_steps - 1:
            unmask_prob = 1.0
        else:
            unmask_prob = (t_val - t_next_val) / t_val

        unmask_flags = jax.random.bernoulli(rng_unmask, p=unmask_prob, shape=(1, seq_len))
        should_unmask = is_masked & unmask_flags

        # 3. Cập nhật chuỗi
        x = jnp.where(should_unmask, x_0_pred, x)

        # In trạng thái trung gian mỗi vài bước
        masked_count = int(jnp.sum(x == tokenizer.mask_id))
        pct_masked = (masked_count / seq_len) * 100.0
        
        current_text = tokenizer.decode(list(x[0]))
        step_num = step + 1
        print(f"Bước [{step_num:02d}/{num_steps:02d}] (t={t_next_val:.2f} | Còn {masked_count:02d} masks - {pct_masked:4.1f}%):")
        print(f"  {current_text}")

    print("\n" + "=" * 75)
    print(" KẾT QUẢ CUỐI CÙNG HOÀN TOÀN KHỬ NHIỄU:")
    print(f"  \"{tokenizer.decode(list(x[0]))}\"")
    print("=" * 75)


if __name__ == "__main__":
    # Minh họa nhanh với trọng số ngẫu nhiên
    tokenizer = CharTokenizer()
    rng = jax.random.PRNGKey(42)
    rng_init, rng_sample = jax.random.split(rng)
    
    params = init_transformer_params(
        rng_init,
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        max_len=64
    )
    visualize_sampling_trajectory(params, tokenizer, rng_sample, seq_len=64, num_steps=10)
