import numpy as np
from typing import Literal, Callable, Tuple, Dict
from scipy.stats import norm
import warnings
from numba import jit, prange
import time


class AmericanOptionPricer:
    """
    美式期权定价器（使用最小二乘蒙特卡洛方法）
    支持看涨/看跌期权，包含希腊字母计算
    """

    def __init__(self, seed: int = 42):
        """
        初始化定价器

        参数:
        seed: 随机种子，确保结果可重现
        """
        self.seed = seed
        np.random.seed(seed)

    def simulate_paths_vectorized(
            self,
            S: float,
            r: float,
            sigma: float,
            dt: float,
            n_simulations: int,
            n_steps: int
    ) -> np.ndarray:
        """
        使用向量化方法模拟资产价格路径（几何布朗运动）

        参数:
        S: 初始股票价格
        r: 无风险利率
        sigma: 波动率
        dt: 时间步长
        n_simulations: 模拟路径数量
        n_steps: 时间步数

        返回:
        paths: 形状为 (n_simulations, n_steps + 1) 的数组
        """
        # 生成随机数
        z = np.random.normal(0, 1, (n_simulations, n_steps))

        # 计算每日收益率
        returns = np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

        # 计算累积收益
        cumulative_returns = np.cumprod(returns, axis=1)

        # 构建价格路径
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S
        paths[:, 1:] = S * cumulative_returns

        return paths

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def simulate_paths_fast(
            S: float,
            r: float,
            sigma: float,
            dt: float,
            n_simulations: int,
            n_steps: int
    ) -> np.ndarray:
        """
        使用Numba加速的路径模拟（更快的实现）
        """
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S

        for i in prange(n_simulations):
            for j in range(1, n_steps + 1):
                z = np.random.normal(0, 1)
                paths[i, j] = paths[i, j - 1] * np.exp(
                    (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
                )
        return paths

    def payoff_function(
            self,
            S: np.ndarray,
            K: float,
            option_type: Literal["call", "put"]
    ) -> np.ndarray:
        """
        计算期权收益

        参数:
        S: 股票价格数组
        K: 执行价格
        option_type: 期权类型

        返回:
        收益数组
        """
        if option_type == "call":
            return np.maximum(S - K, 0)
        elif option_type == "put":
            return np.maximum(K - S, 0)
        else:
            raise ValueError("期权类型必须是 'call' 或 'put'")

    def lsmc_price(
            self,
            S: float,
            K: float,
            r: float,
            T: float,
            sigma: float,
            n_simulations: int = 10000,
            n_steps: int = 100,
            option_type: Literal["call", "put"] = "put",
            use_antithetic: bool = True,
            regression_basis: int = 3,
            use_fast_simulation: bool = False
    ) -> Tuple[float, Dict]:
        """
        使用最小二乘蒙特卡洛(LSMC)方法计算美式期权价格

        参数:
        S: 初始股票价格
        K: 执行价格
        r: 无风险利率
        T: 到期时间
        sigma: 波动率
        n_simulations: 模拟路径数量
        n_steps: 时间步数
        option_type: 期权类型
        use_antithetic: 是否使用对偶变量法减少方差
        regression_basis: 回归基函数的最高次数
        use_fast_simulation: 是否使用Numba加速的模拟

        返回:
        option_price: 期权价格
        info: 包含额外信息的字典
        """
        # 验证参数
        if n_simulations < 1000:
            warnings.warn("模拟路径数较少，结果可能不稳定")

        dt = T / n_steps
        discount_factor = np.exp(-r * dt)

        # 模拟价格路径
        if use_antithetic:
            # 使用对偶变量法减少方差
            n_half = n_simulations // 2
            if use_fast_simulation:
                paths1 = self.simulate_paths_fast(S, r, sigma, dt, n_half, n_steps)
                paths2 = self.simulate_paths_fast(S, r, sigma, dt, n_half, n_steps)
            else:
                paths1 = self.simulate_paths_vectorized(S, r, sigma, dt, n_half, n_steps)
                # 生成对偶路径
                paths2 = 2 * S - paths1
            paths = np.vstack([paths1, paths2])
        else:
            if use_fast_simulation:
                paths = self.simulate_paths_fast(S, r, sigma, dt, n_simulations, n_steps)
            else:
                paths = self.simulate_paths_vectorized(S, r, sigma, dt, n_simulations, n_steps)

        # 初始化现金流矩阵
        cash_flows = np.zeros_like(paths)
        cash_flows[:, -1] = self.payoff_function(paths[:, -1], K, option_type)

        # 向后归纳
        for t in range(n_steps - 1, 0, -1):
            # 获取当前时刻的价格
            S_t = paths[:, t]

            # 计算立即行权收益
            immediate_payoff = self.payoff_function(S_t, K, option_type)

            # 只有在立即行权收益大于0时才考虑回归
            in_the_money = immediate_payoff > 0

            if np.sum(in_the_money) > 0:
                # 获取实值路径
                S_in_money = S_t[in_the_money]

                # 计算未来现金流的现值
                future_cf = cash_flows[in_the_money, t + 1:].sum(axis=1)
                continuation_value = future_cf * np.exp(-r * dt)

                # 构建回归矩阵
                X = np.column_stack([S_in_money ** i for i in range(regression_basis + 1)])

                # 最小二乘回归
                try:
                    beta = np.linalg.lstsq(X, continuation_value, rcond=None)[0]
                    estimated_continuation = X @ beta
                except:
                    # 如果回归失败，使用简单平均值
                    estimated_continuation = np.full_like(continuation_value, np.mean(continuation_value))

                # 比较立即行权和继续持有价值
                exercise = immediate_payoff[in_the_money] > estimated_continuation

                # 更新现金流
                exercise_indices = np.where(in_the_money)[0]
                cash_flows[exercise_indices[exercise], t] = immediate_payoff[exercise_indices[exercise]]
                cash_flows[exercise_indices[exercise], t + 1:] = 0

        # 计算期权价格
        all_cash_flows = cash_flows.sum(axis=1)
        present_values = all_cash_flows * np.exp(-r * dt)

        # 计算标准误
        option_price = np.mean(present_values)
        standard_error = np.std(present_values) / np.sqrt(n_simulations)
        confidence_interval = (
            option_price - 1.96 * standard_error,
            option_price + 1.96 * standard_error
        )

        # 返回结果
        info = {
            "standard_error": standard_error,
            "confidence_interval": confidence_interval,
            "paths_simulated": n_simulations,
            "time_steps": n_steps,
            "dt": dt
        }

        return option_price, info

    def calculate_greeks(
            self,
            S: float,
            K: float,
            r: float,
            T: float,
            sigma: float,
            n_simulations: int = 10000,
            n_steps: int = 100,
            option_type: Literal["call", "put"] = "put",
            bump_size: float = 0.01
    ) -> Dict[str, float]:
        """
        使用扰动法计算希腊字母

        参数:
        S, K, r, T, sigma: 期权参数
        n_simulations, n_steps: 模拟参数
        option_type: 期权类型
        bump_size: 扰动大小

        返回:
        greeks: 希腊字母字典
        """
        # 基准价格
        base_price, _ = self.lsmc_price(
            S, K, r, T, sigma, n_simulations, n_steps, option_type
        )

        # 1. Delta: 价格对标的资产价格的敏感度
        price_up, _ = self.lsmc_price(
            S * (1 + bump_size), K, r, T, sigma, n_simulations, n_steps, option_type
        )
        price_down, _ = self.lsmc_price(
            S * (1 - bump_size), K, r, T, sigma, n_simulations, n_steps, option_type
        )
        delta = (price_up - price_down) / (2 * S * bump_size)

        # 2. Gamma: Delta对标的资产价格的二阶敏感度
        gamma = (price_up - 2 * base_price + price_down) / (S ** 2 * bump_size ** 2)

        # 3. Vega: 价格对波动率的敏感度
        price_vega_up, _ = self.lsmc_price(
            S, K, r, T, sigma + bump_size, n_simulations, n_steps, option_type
        )
        price_vega_down, _ = self.lsmc_price(
            S, K, r, T, sigma - bump_size, n_simulations, n_steps, option_type
        )
        vega = (price_vega_up - price_vega_down) / (2 * bump_size)

        # 4. Rho: 价格对无风险利率的敏感度
        price_rho_up, _ = self.lsmc_price(
            S, K, r + bump_size, T, sigma, n_simulations, n_steps, option_type
        )
        price_rho_down, _ = self.lsmc_price(
            S, K, r - bump_size, T, sigma, n_simulations, n_steps, option_type
        )
        rho = (price_rho_up - price_rho_down) / (2 * bump_size)

        # 5. Theta: 价格对时间的敏感度（每天变化）
        if T > bump_size / 365:  # 确保不会出现负时间
            price_theta, _ = self.lsmc_price(
                S, K, r, T - bump_size / 365, sigma, n_simulations, n_steps, option_type
            )
            theta = (price_theta - base_price) / (bump_size / 365)
        else:
            # 如果时间太短，使用近似
            theta = 0

        greeks = {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho,
            "theta": theta
        }

        return greeks

    def calculate_greeks_fast(
            self,
            S: float,
            K: float,
            r: float,
            T: float,
            sigma: float,
            n_simulations: int = 5000,  # 减少模拟次数以加速
            n_steps: int = 50,  # 减少步数以加速
            option_type: Literal["call", "put"] = "put",
            bump_size: float = 0.01
    ) -> Dict[str, float]:
        """
        快速计算希腊字母（使用较少的模拟次数）
        """
        return self.calculate_greeks(
            S, K, r, T, sigma, n_simulations, n_steps, option_type, bump_size
        )


# 示例使用
if __name__ == "__main__":
    # 设置参数
    S = 100  # 初始股票价格
    K = 100  # 执行价格
    r = 0.05  # 无风险利率
    T = 1.0  # 到期时间（年）
    sigma = 0.2  # 波动率
    n_simulations = 10000  # 模拟路径数
    n_steps = 100  # 时间步数
    option_type = "put"  # 期权类型

    # 创建定价器
    pricer = AmericanOptionPricer(seed=42)

    print("=" * 60)
    print("美式期权定价 (LSMC方法)")
    print("=" * 60)
    print(f"参数设置:")
    print(f"  初始价格 S = {S}")
    print(f"  执行价格 K = {K}")
    print(f"  无风险利率 r = {r:.1%}")
    print(f"  到期时间 T = {T} 年")
    print(f"  波动率 σ = {sigma:.1%}")
    print(f"  期权类型 = {option_type}")
    print(f"  模拟路径数 = {n_simulations:,}")
    print(f"  时间步数 = {n_steps}")
    print("-" * 60)

    # 计算期权价格
    start_time = time.time()
    price, info = pricer.lsmc_price(
        S, K, r, T, sigma, n_simulations, n_steps, option_type
    )
    elapsed_time = time.time() - start_time

    print(f"期权价格: {price:.4f}")
    print(f"标准误差: {info['standard_error']:.4f}")
    print(f"95% 置信区间: [{info['confidence_interval'][0]:.4f}, {info['confidence_interval'][1]:.4f}]")
    print(f"计算时间: {elapsed_time:.2f} 秒")
    print("-" * 60)

    # 计算希腊字母
    print("\n计算希腊字母...")
    greeks = pricer.calculate_greeks_fast(
        S, K, r, T, sigma, n_simulations=5000, option_type=option_type
    )

    print("希腊字母:")
    print(f"  Delta (Δ): {greeks['delta']:.4f}")
    print(f"  Gamma (Γ): {greeks['gamma']:.6f}")
    print(f"  Vega  (ν): {greeks['vega']:.4f}")
    print(f"  Rho   (ρ): {greeks['rho']:.4f}")
    print(f"  Theta (Θ): {greeks['theta']:.4f} (每日)")

    # 对比：美式 vs 欧式看跌期权价格
    print("\n" + "=" * 60)
    print("美式 vs 欧式期权对比")
    print("=" * 60)


    def european_put_price(S, K, r, T, sigma):
        """计算欧式看跌期权价格 (Black-Scholes)"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


    european_price = european_put_price(S, K, r, T, sigma)
    print(f"欧式看跌期权价格: {european_price:.4f}")
    print(f"美式看跌期权价格: {price:.4f}")
    print(f"提前行权溢价: {price - european_price:.4f}")
    print(f"溢价比例: {(price - european_price) / european_price:.2%}")

    # 敏感性分析示例
    print("\n" + "=" * 60)
    print("敏感性分析示例")
    print("=" * 60)

    # 不同波动率下的价格
    print("\n不同波动率下的美式期权价格:")
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
    for sig in sigmas:
        p, _ = pricer.lsmc_price(S, K, r, T, sig, 5000, 50, "put", use_fast_simulation=True)
        print(f"  σ={sig:.1%}: {p:.4f}")

    # 不同到期时间下的价格
    print("\n不同到期时间下的美式期权价格:")
    times = [0.25, 0.5, 1.0, 2.0, 3.0]
    for t in times:
        p, _ = pricer.lsmc_price(S, K, r, t, sigma, 5000, 50, "put", use_fast_simulation=True)
        print(f"  T={t:.2f} 年: {p:.4f}")