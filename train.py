"""
================================================================================
SCRIPT HUẤN LUYỆN: MINI DIFFUSION LANGUAGE MODEL VỚI TỐI ƯU HÓA MUON (PURE JAX)
================================================================================
Kịch bản huấn luyện hoàn chỉnh, tích hợp mọi thành phần đã xây dựng:
1. Mô hình Bidirectional Transformer với điều kiện hóa bước thời gian (Time Conditioning).
2. Quy trình khuếch tán rời rạc (Discrete Masked Diffusion - MDLM).
3. Trình tối ưu hóa Muon (Newton-Schulz Polar Decomposition) kết hợp AdamW.
4. Lập lịch tốc độ học Cosine Warmup.
5. Biên dịch hàm toàn vẹn từ Gradient đến Update bằng `jax.jit`.

TẠI SAO `jax.value_and_grad` VƯỢT TRỘI HƠN PYTORCH TRONG THIẾT KẾ GIẢNG DẠY?
--------------------------------------------------------------------------
Trong PyTorch, ta gọi `loss.backward()`, các gradient được tích lũy ngầm vào thuộc
tính `.grad` của từng Tensor (stateful).
Trong JAX:
  `grad_fn = jax.grad(loss_fn)`
  `grads = grad_fn(params)`
Gradient là một giá trị toán học thuần khiết (Pure Value), có cấu trúc PyTree y hệt
`params`. `jax.value_and_grad(loss_fn, has_aux=True)` vừa trả về giá trị hàm mất mát,
các chỉ số phụ (như độ chính xác), vừa trả về toàn bộ gradient trong một lần duyệt
đồ thị tính toán (Reverse-mode Automatic Differentiation) tối ưu nhất!
"""

import time
import math
import jax
import jax.numpy as jnp

from src.tokenizer import CharTokenizer
from src.model import init_transformer_params
from src.diffusion import compute_loss, sample_tokens
from src.optimizers.muon import init_muon_state, muon_step
from src.optimizers.schedulers import cosine_warmup_schedule
from src.utils import print_model_summary, prepare_dataset


# ==============================================================================
# TẬP DỮ LIỆU MẪU DÀNH CHO HUẤN LUYỆN NHANH (EDUCATIONAL TOY CORPUS)
# ==============================================================================
# Một đoạn văn bản triết học & khoa học máy tính đa dạng để mô hình học cấu trúc câu
TOY_CORPUS = """
Artificial intelligence and deep learning have transformed the way we understand computation and cognition.
Diffusion models represent a paradigm shift in generative modeling, moving away from simple autoregressive next token prediction.
By learning to reverse a continuous or discrete corruption process, diffusion models can plan and generate coherent sequences bidirectionally.
The Muon optimizer leverages Newton Schulz iterations to compute orthogonal updates for large weight matrices.
Orthogonalized momentum maintains the spectral energy across all subspace directions, preventing gradient collapse.
Pure functional programming in JAX provides mathematical clarity, deterministic reproducibility, and accelerated execution through XLA compilation.
Language models understand grammar, syntax, semantics, and context by attending to bidirectional representations.
In masked diffusion models, tokens transition into an absorbing mask state during the forward noising process.
The neural network learns to denoise and recover the original words from corrupted observations.
Through iterative ancestral sampling, order emerges from pure noise step by step.
Linear warmup stabilizes the initial noisy gradients before cosine annealing guides convergence toward optimal parameter basins.
"""


def main():
    print("=" * 75)
    print(" BẮT ĐẦU CHƯƠNG TRÌNH HUẤN LUYỆN JAX DIFFUSION LM VỚI MUON OPTIMIZER")
    print("=" * 75)

    # 1. Khởi tạo thiết bị và hạt giống ngẫu nhiên
    devices = jax.devices()
    print(f"Thiết bị tính toán JAX: {devices}")
    
    master_key = jax.random.PRNGKey(2026)
    master_key, k_init, k_train = jax.random.split(master_key, 3)

    # 2. Xây dựng Tokenizer và nạp dữ liệu
    tokenizer = CharTokenizer([TOY_CORPUS])
    print(f"Kích thước từ vựng ký tự (Vocab Size): {tokenizer.vocab_size} tokens")
    print(f"Token đặc biệt: MASK_ID={tokenizer.mask_id}, PAD_ID={tokenizer.pad_id}")

    # Lặp lại văn bản để tạo tập dữ liệu đủ lớn cho các bước cập nhật
    full_text = TOY_CORPUS * 50
    seq_len = 64
    dataset = prepare_dataset(full_text, tokenizer, seq_len=seq_len)
    num_samples = dataset.shape[0]
    print(f"Đã chuẩn bị {num_samples} mẫu câu huấn luyện với độ dài cố định L={seq_len} ký tự.")

    # 3. Cấu hình siêu tham số mô hình (Hyperparameters)
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 512
    batch_size = 16
    total_steps = 250
    warmup_steps = 25
    base_lr = 2e-3
    weight_decay = 0.01

    print("\n--- Khởi tạo tham số mô hình thuần JAX ---")
    params = init_transformer_params(
        k_init,
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=seq_len
    )
    print_model_summary(params)

    # 4. Khởi tạo trạng thái bộ tối ưu Muon + AdamW
    print("\n--- Khởi tạo trạng thái Muon Optimizer (Newton-Schulz) ---")
    opt_state = init_muon_state(params)
    print(f"Trạng thái optimizer ban đầu tại step: {opt_state.step}")

    # ==========================================================================
    # ĐỊNH NGHĨA BƯỚC HUẤN LUYỆN THUẦN KHIẾT (COMPILED JAX JIT STEP)
    # ==========================================================================
    # Toàn bộ phép tính: Lan truyền tiến -> Tính Loss -> Tự động tính vi phân (Grad)
    # -> Cập nhật Newton-Schulz cho ma trận 2D -> Cập nhật AdamW cho 1D
    # đều được XLA nung thành một chuỗi Kernel GPU/Metal duy nhất!
    @jax.jit
    def train_step(params, opt_state, batch, rng, lr):
        """
        Hàm thực hiện một bước huấn luyện nguyên tử (Atomic Training Step).
        """
        # Định nghĩa hàm loss bao đóng nhận params làm biến số vi phân
        def loss_wrapper(p):
            return compute_loss(
                p,
                rng=rng,
                x_0=batch,
                mask_token_id=tokenizer.mask_id,
                pad_token_id=tokenizer.pad_id,
                num_heads=num_heads
            )

        # Tính đạo hàm riêng toàn bộ ma trận trọng số trong 1 lệnh duy nhất
        (loss, metrics), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)

        # Cập nhật trọng số qua thuật toán Muon
        new_params, new_opt_state = muon_step(
            params=params,
            grads=grads,
            state=opt_state,
            lr=lr,
            weight_decay=weight_decay,
            ns_steps=5
        )

        return new_params, new_opt_state, metrics

    # 5. Vòng lặp huấn luyện chính (Training Loop)
    print("\n--- Bắt đầu vòng lặp huấn luyện (Training Loop) ---")
    start_time = time.time()
    
    num_batches = num_samples // batch_size

    for step in range(1, total_steps + 1):
        k_train, k_step, k_sample = jax.random.split(k_train, 3)

        # Lấy mini-batch xoay vòng đơn giản
        batch_idx = (step - 1) % num_batches
        batch = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # Tính learning rate theo lịch Cosine Warmup
        current_lr = cosine_warmup_schedule(
            step=step,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        # Chạy bước huấn luyện tăng tốc XLA
        params, opt_state, metrics = train_step(params, opt_state, batch, k_step, current_lr)

        # In nhật ký huấn luyện định kỳ
        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start_time
            loss_val = float(metrics["loss"])
            acc_val = float(metrics["masked_acc"]) * 100.0
            ratio_val = float(metrics["masked_ratio"]) * 100.0
            print(
                f"Bước [{step:03d}/{total_steps:03d}] | "
                f"Loss: {loss_val:.4f} | "
                f"Masked Acc: {acc_val:5.1f}% | "
                f"Tỷ lệ che: {ratio_val:4.1f}% | "
                f"LR: {float(current_lr):.6f} | "
                f"Thời gian: {elapsed:.1f}s"
            )

        # 6. Trình diễn quá trình sinh câu (Denoising Generation) mỗi 50 bước
        if step % 50 == 0:
            print("\n" + "-" * 60)
            print(f" TRÌNH DIỄN SINH KHUẾCH TÁN (DIFFUSION SAMPLING) TẠI BƯỚC {step}:")
            print("-" * 60)
            generated_ids = sample_tokens(
                params=params,
                rng=k_sample,
                batch_size=1,
                seq_len=seq_len,
                num_steps=32,
                mask_token_id=tokenizer.mask_id,
                num_heads=num_heads,
                temperature=0.8
            )
            generated_text = tokenizer.decode(list(generated_ids[0]))
            print(f"Văn bản sinh ra:\n\"{generated_text}\"")
            print("-" * 60 + "\n")

    total_time = time.time() - start_time
    print("=" * 75)
    print(f" HOÀN THÀNH HUẤN LUYỆN! Tổng thời gian chạy: {total_time:.2f} giây.")
    print("=" * 75)


if __name__ == "__main__":
    main()
