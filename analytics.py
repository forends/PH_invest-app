import numpy as np
import pandas as pd
import streamlit as st


def calculate_metrics(returns, risk_free=0.03):
    # CAGR
    cumulative = (1 + returns).prod()
    years = len(returns) / 252
    cagr = cumulative ** (1 / years) - 1

    # Volatility
    vol = returns.std() * np.sqrt(252)

    # Sharpe
    sharpe = (cagr - risk_free) / vol if vol != 0 else 0

    # MDD
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()

    return cagr, vol, sharpe, mdd


def calculate_var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)


def contribution(weights, returns):
    return weights * returns.mean()


# ---------------- UI ---------------- #

st.header("📊 고급 투자 분석")

# 예시 데이터 (나중에 실제 데이터 연결)
tickers = ["AAPL", "MSFT", "NVDA"]
weights = np.array([0.3, 0.4, 0.3])

# 가짜 수익률
data = pd.DataFrame(
    np.random.normal(0.001, 0.02, (252, len(tickers))),
    columns=tickers,
)

portfolio_returns = (data * weights).sum(axis=1)

cagr, vol, sharpe, mdd = calculate_metrics(portfolio_returns)
var95 = calculate_var(portfolio_returns)
contri = contribution(weights, data)

# ----------- 출력 ----------- #

col1, col2, col3, col4 = st.columns(4)

col1.metric("CAGR", f"{cagr*100:.2f}%")
col2.metric("변동성", f"{vol*100:.2f}%")
col3.metric("Sharpe", f"{sharpe:.2f}")
col4.metric("MDD", f"{mdd*100:.2f}%")

st.subheader("📉 VaR (95%)")
st.write(f"하루 최대 손실 가능성: {var95*100:.2f}%")

st.subheader("📌 종목별 수익 기여도")
for t, c in zip(tickers, contri):
    st.write(f"{t}: {c*100:.2f}%")