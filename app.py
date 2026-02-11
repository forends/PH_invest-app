import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("Professional Portfolio System")

# =====================================================
# 투자 유니버스
# =====================================================
STOCK_UNIVERSE = [
    "SPY","QQQ","VTI","IWM","VEA","VWO",
    "TLT","IEF","GLD",
    "AAPL","MSFT","NVDA","AMZN","GOOGL"
]

# =====================================================
# 데이터 로드
# =====================================================
@st.cache_data
def load_price(tickers):
    df = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")

# =====================================================
# 포트폴리오 생성
# =====================================================
def generate_portfolio():
    picks = random.sample(STOCK_UNIVERSE, 8)
    weights = np.random.dirichlet(np.ones(len(picks)), size=1)[0]
    return picks, weights

if "picks" not in st.session_state:
    st.session_state.picks, st.session_state.weights = generate_portfolio()

picks = st.session_state.picks
weights = st.session_state.weights

prices = load_price(picks)

# =====================================================
# 수익률 계산
# =====================================================
returns = prices.pct_change().dropna()

mean_returns = returns.mean() * 252
cov = returns.cov() * 252

# 기본 기대 수익 & 변동성
exp_return = float(np.dot(weights, mean_returns) * 100)
volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * 100)

# =====================================================
# 누적 수익률 (백테스트 기반)
# =====================================================
port_daily = returns.dot(weights)
cum = (1 + port_daily).cumprod()

# =====================================================
# 📈 프로 성과 지표
# =====================================================

# CAGR
days = len(cum)
cagr = (cum.iloc[-1] ** (252/days) - 1) * 100

# Sharpe Ratio (무위험 수익률 2% 가정)
rf = 0.02
sharpe = (port_daily.mean()*252 - rf) / (port_daily.std()*np.sqrt(252))

# MDD
rolling_max = cum.cummax()
drawdown = cum / rolling_max - 1
mdd = drawdown.min() * 100

# =====================================================
# 레이아웃
# =====================================================
left, right = st.columns([3,1])

# =====================================================
# 좌측 : 포트폴리오 분석
# =====================================================
with left:
    st.subheader("Performance Dashboard")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected Return", f"{exp_return:.2f}%")
    k2.metric("Volatility", f"{volatility:.2f}%")
    k3.metric("CAGR", f"{cagr:.2f}%")
    k4.metric("Sharpe", f"{sharpe:.2f}")

    st.line_chart(cum)

    st.caption(f"Maximum Drawdown (MDD) : {mdd:.2f}%")

# =====================================================
# 우측 : 종목 & 비중
# =====================================================
with right:
    st.subheader("Portfolio")

    df = pd.DataFrame({
        "Ticker": picks,
        "Weight(%)": [round(w*100,2) for w in weights]
    })

    st.dataframe(df, use_container_width=True)

    if st.button("AI 전략 다시 계산"):
        st.session_state.picks, st.session_state.weights = generate_portfolio()
        st.cache_data.clear()
        st.rerun()