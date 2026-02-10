import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# =====================================================
# 포트폴리오 정의
# =====================================================
PORT_INFO = {
    "SPY": {"weight": 0.10, "reason": "US Large Cap"},
    "QQQ": {"weight": 0.15, "reason": "Nasdaq Growth"},
    "VTI": {"weight": 0.10, "reason": "Total Market"},

    "TQQQ": {"weight": 0.15, "reason": "Leveraged Nasdaq"},
    "UPRO": {"weight": 0.10, "reason": "Leveraged S&P"},
    "TECL": {"weight": 0.10, "reason": "Tech Leveraged"},

    "SMH": {"weight": 0.10, "reason": "Semiconductor"},
    "BOTZ": {"weight": 0.05, "reason": "AI & Robotics"},
    "SKYY": {"weight": 0.05, "reason": "Cloud"},

    "SCHD": {"weight": 0.07, "reason": "Dividend Quality"},
    "TLT": {"weight": 0.03, "reason": "Long Treasury"}
}

TICKERS = list(PORT_INFO.keys())

# =====================================================
# 데이터 다운로드
# =====================================================
@st.cache_data(ttl=3600)
def load_data(tickers):
    data = yf.download(tickers, period="1y", auto_adjust=True)
    return data

prices = load_data(TICKERS)

if prices.empty:
    st.stop()

# =====================================================
# 수익률 계산
# =====================================================
returns = prices.pct_change().dropna()

# MultiIndex 제거
if isinstance(returns.columns, pd.MultiIndex):
    returns.columns = returns.columns.get_level_values(-1)

exp_returns = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)

# =====================================================
# 포트폴리오 기대값 계산
# =====================================================
weights = np.array([PORT_INFO[t]["weight"] for t in TICKERS])

port_return = sum(exp_returns[t] * PORT_INFO[t]["weight"] for t in TICKERS) * 100
port_vol = sum(volatility[t] * PORT_INFO[t]["weight"] for t in TICKERS) * 100

# =====================================================
# 상단 KPI
# =====================================================
st.title("Portfolio Strategy Dashboard")

k1, k2, k3 = st.columns(3)
k1.metric("Expected Return (1Y)", f"{port_return:.2f}%")
k2.metric("Volatility (1Y)", f"{port_vol:.2f}%")
k3.metric("Number of Assets", len(TICKERS))

st.divider()

# =====================================================
# 메인 좌/우
# =====================================================
left, right = st.columns([2, 1])

# =====================================================
# 자산 구성 테이블
# =====================================================
with left:
    st.subheader("Asset Allocation")

    table = pd.DataFrame({
        "Ticker": TICKERS,
        "Weight": [PORT_INFO[t]["weight"] * 100 for t in TICKERS],
        "Exp Return": [exp_returns[t] * 100 for t in TICKERS],
        "Volatility": [volatility[t] * 100 for t in TICKERS],
        "Role": [PORT_INFO[t]["reason"] for t in TICKERS],
    })

    st.dataframe(
        table.style.format({
            "Weight": "{:.1f}%",
            "Exp Return": "{:.1f}%",
            "Volatility": "{:.1f}%"
        }),
        use_container_width=True
    )

# =====================================================
# 리스크 & 리밸런싱
# =====================================================
with right:
    st.subheader("Risk Monitor")

    drift = np.abs(weights - weights.mean())

    if drift.max() > 0.08:
        st.error("Rebalancing Required")
    else:
        st.success("Allocation Stable")

    st.divider()

    st.subheader("Weight Distribution")
    weight_df = pd.DataFrame({"weight": weights}, index=TICKERS)
    st.bar_chart(weight_df)

# =====================================================
# 성과 차트
# =====================================================
st.divider()
st.subheader("Cumulative Performance (1Y)")

cum = (1 + returns).cumprod()

# MultiIndex 방어
if isinstance(cum.columns, pd.MultiIndex):
    cum.columns = cum.columns.get_level_values(-1)

st.line_chart(cum)
