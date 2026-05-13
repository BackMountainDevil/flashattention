import numpy as np
import torch
import torch.nn.functional as F


def flash_attention_std_np(q, k, v):
    """
    标准自注意力实现（缩放点积注意力）
    q: [B, N, S, D]
    k: [B, N, S, D]
    v: [B, N, S, D]
    return: out [B, N, S, D]
    """
    B, N, S, D = q.shape
    # 缩放因子
    scale = 1.0 / np.sqrt(D)

    # (B, N, S, D) @ (B, N, D, S) -> (B, N, S, S)
    attn = np.matmul(q * scale, k.transpose(0, 1, 3, 2))

    # Softmax
    attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))  # 防溢出
    attn = attn / np.sum(attn, axis=-1, keepdims=True)

    # (B, N, S, S) @ (B, N, S, D) -> (B, N, S, D)
    out = np.matmul(attn, v)
    return out


def flash_attention_std_torch(q, k, v, attn_mask_np=None):
    """
    PyTorch 官方标准 scaled_dot_product_attention
    输入输出都是 numpy 数组，内部转 tensor 计算
    """
    q_tensor = torch.from_numpy(q)
    k_tensor = torch.from_numpy(k)
    v_tensor = torch.from_numpy(v)
    if not attn_mask_np is None:
        # 把【分块 mask】转为 torch 需要的【稠密矩阵 mask】
        # 稠密 mask shape: [B, N, Sq, Skv]
        B, N, S, _ = q.shape
        dense_mask = np.zeros((B, N, S, S), dtype=bool)
        for b in range(B):
            for h in range(N):
                for i in range(num_q_blocks):
                    q_s = i * block_size_q
                    q_e = min(q_s + block_size_q, S)
                    for j in range(num_kv_blocks):
                        kv_s = j * block_size_kv
                        kv_e = min(kv_s + block_size_kv, S)
                        dense_mask[b, h, q_s:q_e, kv_s:kv_e] = attn_mask_np[b, h, i, j]
        with torch.no_grad():
            out = F.scaled_dot_product_attention(
                q_tensor, k_tensor, v_tensor, attn_mask=torch.from_numpy(dense_mask)
            )
    else:
        with torch.no_grad():
            out = F.scaled_dot_product_attention(q_tensor, k_tensor, v_tensor)
    return out.numpy()


def flash_attention_v2_cpu(
    q, k, v, block_size_q=128, block_size_kv=128, attn_mask=None
):
    """
    CPU 模拟 FlashAttention-2 分块自注意力
    严格遵循分块 Q、分块 KV、累加输出、Online Softmax 思想
    支持不同的 Q 和 KV 块大小
    支持块级 Mask: attn_mask shape: [B, N, num_q_blocks, num_kv_blocks]
    """
    B, N, S, D = q.shape
    scale = 1.0 / np.sqrt(D)

    # 输出初始化
    out = np.zeros_like(q)

    for b in range(B):  # batch 循环
        for h in range(N):  # head 循环
            Q = q[b, h]  # [S, D]
            K = k[b, h]  # [S, D]
            V = v[b, h]  # [S, D]

            # Q 分块数
            num_q_blocks = (S + block_size_q - 1) // block_size_q

            for i in range(num_q_blocks):
                # Q 块起始/结束
                q_start = i * block_size_q
                q_end = min(q_start + block_size_q, S)
                q_tile = Q[q_start:q_end, :]  # [Br, D]

                # 存储当前 Q 块的输出累加、softmax 分母、最大值（防溢出）
                O_i = np.zeros((q_end - q_start, D))
                l_i = np.zeros((q_end - q_start, 1))
                m_i = np.ones((q_end - q_start, 1)) * -np.inf

                # 遍历所有 KV 块
                num_kv_blocks = (S + block_size_kv - 1) // block_size_kv
                for j in range(num_kv_blocks):
                    if attn_mask is not None and not attn_mask[b, h, i, j]:
                        # 如果 mask 为 False，表示该块不参与计算，直接跳过
                        continue

                    kv_start = j * block_size_kv
                    kv_end = min(kv_start + block_size_kv, S)
                    k_tile = K[kv_start:kv_end, :]  # [Bc, D]
                    v_tile = V[kv_start:kv_end, :]  # [Bc, D]

                    # 计算 S_ij = Q_i @ K_j.T / sqrt(D)
                    s_ij = np.matmul(q_tile, k_tile.T) * scale

                    # 在线 softmax 核心：更新全局 max 防止指数溢出
                    m_i_new = np.maximum(m_i, np.max(s_ij, axis=1, keepdims=True))

                    # 计算修正系数
                    exp_old = np.exp(m_i - m_i_new)
                    exp_new = np.exp(s_ij - m_i_new)

                    # 新的分母
                    l_i_new = exp_old * l_i + np.sum(exp_new, axis=1, keepdims=True)

                    # 更新输出
                    O_i = O_i * exp_old  # 修正历史输出
                    O_i = O_i + np.matmul(exp_new, v_tile)

                    # 更新状态
                    l_i = l_i_new
                    m_i = m_i_new

                # 最终归一化
                # 注意：如果某个 Q 块被完全 mask 掉，l_i 可能为 0，这里做个保护防止除零
                if np.any(l_i != 0):  # 避免除 0
                    O_i = O_i / l_i

                # 写回输出
                out[b, h, q_start:q_end] = O_i

    return out


def compare_error(out1, out2, rtol=1e-5, atol=1e-6):
    """
    比较两个输出的误差
    """
    abs_diff = np.abs(out1 - out2)
    rel_diff = abs_diff / (np.abs(out2) + 1e-8)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)

    # 检查是否在容差范围内
    is_close = np.allclose(out1, out2, rtol=rtol, atol=atol)

    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_rel_diff": mean_rel_diff,
        "is_close": is_close,
    }


if __name__ == "__main__":
    B, N, S, D = 2, 3, 1024, 64  # batch, heads, seq_len, head_dim

    np.random.seed(42)  # 固定随机种子，保证可复现
    q = np.random.randn(B, N, S, D).astype(np.float32)
    k = np.random.randn(B, N, S, D).astype(np.float32)
    v = np.random.randn(B, N, S, D).astype(np.float32)
    # 标准自注意力
    print("正在计算标准自注意力...")
    out_std_np = flash_attention_std_np(q, k, v)
    out_std_torch = flash_attention_std_torch(q, k, v)
    error_stats = compare_error(out_std_np, out_std_torch)
    print(f"在误差范围内: {error_stats['is_close']}")

    # 分块 FlashAttention 模拟
    print("正在计算分块 FlashAttention-v2 (CPU模拟)...")
    out_sim = flash_attention_v2_cpu(q, k, v)

    print("\n===== 误差对比 =====")
    error_stats = compare_error(out_std_torch, out_sim)
    print(f"最大绝对误差: {error_stats['max_abs_diff']:.6e}")
    print(f"最大相对误差: {error_stats['max_rel_diff']:.6e}")
    print(f"平均绝对误差: {error_stats['mean_abs_diff']:.6e}")
    print(f"平均相对误差: {error_stats['mean_rel_diff']:.6e}")
    print(f"是否在误差范围内 {error_stats['is_close']}")
    # 生成【随机分块 mask】
    block_size_q = 128
    block_size_kv = 128
    num_q_blocks = (S + block_size_q - 1) // block_size_q
    num_kv_blocks = (S + block_size_kv - 1) // block_size_kv
    attn_mask_np = np.random.rand(B, N, num_q_blocks, num_kv_blocks) > 0.3
    print("mask shape:", attn_mask_np.shape)
    out_sim = flash_attention_v2_cpu(q, k, v, block_size_q, block_size_kv, attn_mask_np)
    out_std_torch = flash_attention_std_torch(q, k, v, attn_mask_np)
    print("\n===== 误差对比 =====")
    error_stats = compare_error(out_std_torch, out_sim)
    print(f"最大绝对误差: {error_stats['max_abs_diff']:.6e}")
    print(f"最大相对误差: {error_stats['max_rel_diff']:.6e}")
    print(f"平均绝对误差: {error_stats['mean_abs_diff']:.6e}")
    print(f"平均相对误差: {error_stats['mean_rel_diff']:.6e}")
    print(f"是否在误差范围内 {error_stats['is_close']}")
