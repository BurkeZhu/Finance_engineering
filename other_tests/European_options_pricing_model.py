import math
from scipy.stats import norm
from typing import Literal, Union


def black_scholes_option_price(
        S: float,
        K: float,
        r: float,
        T: float,
        sigma: float,
        q: float = 0,
        option_type: Literal["call", "put"] = "call"
) -> float:
    """
    使用Black-Scholes模型计算欧式期权的价格（支持股息调整）

    参数:
    S: 当前股票价格
    K: 执行价格
    r: 无风险利率
    T: 到期时间（年）
    sigma: 波动率
    q: 股息收益率（连续复利），默认为0
    option_type: 期权类型，"call"（看涨）或"put"（看跌），默认为"call"

    返回:
    option_price: 期权价格
    """
    # 计算d1和d2
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # 根据期权类型计算价格
    if option_type == "call":
        option_price = (S * math.exp(-q * T) * norm.cdf(d1) -
                        K * math.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        option_price = (K * math.exp(-r * T) * norm.cdf(-d2) -
                        S * math.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError("无效的期权类型。请选择 'call' 或 'put'。")

    return option_price


# 示例用法
if __name__ == "__main__":
    # 参数设置
    S = 100  # 当前股票价格
    K = 105  # 执行价格
    r = 0.05  # 无风险利率
    T = 0.25  # 到期时间（年）
    sigma = 0.2  # 波动率
    q = 0  # 股息收益率
    option_type = "call"  # 期权类型

    # 计算期权价格
    option_price = black_scholes_option_price(S, K, r, T, sigma, q, option_type)
    print(f"期权价格: {option_price:.4f}")

    # 额外示例：计算看跌期权价格
    put_price = black_scholes_option_price(S, K, r, T, sigma, q, "put")
    print(f"看跌期权价格: {put_price:.4f}")


    # 计算希腊字母示例
    def black_scholes_greeks(S, K, r, T, sigma, q=0, option_type="call"):
        """计算Black-Scholes希腊字母"""
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        greeks = {}

        if option_type == "call":
            greeks["delta"] = math.exp(-q * T) * norm.cdf(d1)
            greeks["gamma"] = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
            greeks["theta"] = - (S * sigma * math.exp(-q * T) * norm.pdf(d1)) / (2 * math.sqrt(T)) \
                              - r * K * math.exp(-r * T) * norm.cdf(d2) \
                              + q * S * math.exp(-q * T) * norm.cdf(d1)
        else:  # put
            greeks["delta"] = math.exp(-q * T) * (norm.cdf(d1) - 1)
            greeks["gamma"] = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
            greeks["theta"] = - (S * sigma * math.exp(-q * T) * norm.pdf(d1)) / (2 * math.sqrt(T)) \
                              + r * K * math.exp(-r * T) * norm.cdf(-d2) \
                              - q * S * math.exp(-q * T) * norm.cdf(-d1)

        greeks["vega"] = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
        greeks["rho"] = K * T * math.exp(-r * T) * norm.cdf(d2) if option_type == "call" \
            else -K * T * math.exp(-r * T) * norm.cdf(-d2)

        return greeks


    # 计算希腊字母
    greeks = black_scholes_greeks(S, K, r, T, sigma, q, option_type)
    print(f"希腊字母: Delta={greeks['delta']:.4f}, Gamma={greeks['gamma']:.4f}, "
          f"Theta={greeks['theta']:.4f}, Vega={greeks['vega']:.4f}, Rho={greeks['rho']:.4f}")