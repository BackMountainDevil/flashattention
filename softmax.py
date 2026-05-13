import numpy as np


def standard_softmax(x):
    """
    标准 Softmax：直接按照公式计算
    公式：e^xi / sum(e^x)
    缺点：当输入值较大时，e^x 极易超出浮点数表示范围（上溢），导致结果为 nan
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def safe_softmax(x):
    """
    Safe Softmax（安全 Softmax）：利用平移不变性解决数值上溢
    公式：e^(xi - max(x)) / sum(e^(x - max(x)))
    原理：将所有输入减去最大值，使得指数部分的最大值为 e^0 = 1，有效防止上溢
    """
    # 减去最大值，保证数值稳定性
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def online_softmax(x):
    """
    Online Softmax（在线 Softmax）：单次遍历，内存高效
    核心思想：不需要提前知道全局最大值，边遍历边动态更新最大值和分母。
    当发现新的最大值时，通过缩放系数修正之前累加的分母。
    这是 FlashAttention 等高效算子节省显存的核心原理。
    """
    # 将输入转换为列表以便迭代
    if isinstance(x, np.ndarray):
        x = x.tolist()

    m = float("-inf")  # 维护当前遇到的最大值
    d = 0.0  # 维护当前的指数和（分母）
    exp_values = []  # 暂存每个元素的指数结果，用于最后计算输出

    # 单次遍历：动态更新最大值和分母
    for xi in x:
        m_new = max(m, xi)
        # 核心步骤：如果最大值更新了，需要将之前累加的分母 d 缩放到新的尺度上
        # 缩放系数为 e^(m_old - m_new)，同时加上当前新值的指数
        d = d * np.exp(m - m_new) + np.exp(xi - m_new)
        exp_values.append(np.exp(xi - m_new))
        m = m_new

    # 计算最终的 Softmax 结果
    return np.array([v / d for v in exp_values])


if __name__ == "__main__":
    # 测试数据1：常规小数值
    x = np.array([1.0, 2.0, 3.0])
    x_large = np.array([1000.0, 2000.0, 3000.0])  # 上溢
    x_small = np.array([-1000, -999, -998])  # 测试下溢

    print("--- 测试常规数值 [1.0, 2.0, 3.0] ---")
    print(f"Standard Softmax: {standard_softmax(x)}")
    print(f"Safe Softmax:     {safe_softmax(x)}")
    print(f"Online Softmax:   {online_softmax(x)}")

    # print("\n--- 测试极大数值 [1000.0, 2000.0, 3000.0] ---")
    # print(f"Standard Softmax: {standard_softmax(x_large)}  <-- 发生上溢，结果为 nan")
    # print(f"Safe Softmax:     {safe_softmax(x_large)}      <-- 数值稳定")
    # print(f"Online Softmax:   {online_softmax(x_large)}    <-- 数值稳定且内存高效")

    print(f"Standard Softmax: {standard_softmax(x_small)}")
    print(f"Safe Softmax:     {safe_softmax(x_small)}")
    print(f"Online Softmax:   {online_softmax(x_small)}")
