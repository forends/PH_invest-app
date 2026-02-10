import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")

st.title("AI Portfolio Advisor")

# ---------------------------------
# 기본 설정
# ---------------------------------
UNIVERSE = [
    "SPY","QQQ","VTI","IWM","VEA","VWO",
    "TLT","IEF","GLD",
    "AAPL","MSFT","NVDA","AMZN","GOOGL"
]

TARGET_RETURN = 10  # %

# ---------------------------------
# 데이터 로드
# ---------------------------------
@st.cache_data
def load_price(tickers):
    df = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")

# ---------------------------------
# 전략 기반 추천 종목 생성
# ---------------------------------
def generate_portfolio():
    picks = random.sample(UNIVERSE, 8)
    weights = np.random.dirichlet(np.ones(len(picks)), size=1)[0]
    return picks, weights

# 세션 상태
if "picks" not in st.session_state:
    st.session_state.picks, st.session_state.weights = generate_portfolio()

picks = st.session_state.picks
weights = st.session_state.weights

prices = load_price(picks)

# ---------------------------------
# 수익률 계산
# ---------------------------------
returns = prices.pct_change().dropna()

mean_returns = returns.mean() * 252
cov = returns.cov() * 252

port_return = float(np.dot(weights, mean_returns) * 100)
port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * 100)

# ---------------------------------
# 위험도 색상
# ---------------------------------
if port_vol < 10:
    risk_color = "🟢 낮음"
elif port_vol < 20:
    risk_color = "🟡 보통"
else:
    risk_color = "🔴 높음"

# ---------------------------------
# 누적 수익률
# ---------------------------------
cum = (1 + returns).cumprod()

# ---------------------------------
# 레이아웃
# ---------------------------------
left, right = st.columns([2,1])

# =================================================
# 좌측 : 포트폴리오 전체 현황
# =================================================
with left:
    st.subheader("Portfolio Overview")

    k1, k2, k3 = st.columns(3)
    k1.metric("Expected Return (1Y)", f"{port_return:.2f}%")
    k2.metric("Volatility", f"{port_vol:.2f}%")
    k3.metric("Risk Level", risk_color)

    st.line_chart(cum)

# =================================================
# 우측 : 종목 / 비중 / 이유 / 알림
# =================================================
with right:
    st.subheader("Recommended Allocation")

    df = pd.DataFrame({
        "Ticker": picks,
        "Weight": weights
    })

    df["Weight"] = (df["Weight"] * 100).round(2)

    # 간단한 추천 이유
    reasons = {
        "SPY":"미국 대형주 대표 ETF",
        "QQQ":"기술주 성장성",
        "VTI":"미국 전체 시장",
        "IWM":"중소형주 분산",
        "VEA":"선진국 분산",
        "VWO":"신흥국 성장",
        "TLT":"금리 하락 대비",
        "IEF":"중기 채권 안정",
        "GLD":"인플레이션 헤지",
        "AAPL":"안정적 실적",
        "MSFT":"클라우드 성장",
        "NVDA":"AI 핵심 수혜",
        "AMZN":"커머스 + 클라우드",
        "GOOGL":"광고 + AI"
    }

    df["Reason"] = df["Ticker"].map(reasons)

    st.dataframe(df, use_container_width=True)

    st.divider()

    # ---------------------------------
    # 목표 수익 알림
    # ---------------------------------
    if port_return >= TARGET_RETURN:
        st.success("🎯 목표 기대수익률 도달!")
    else:
        st.info("목표 수익률 미달 – 성장 자산 확대 가능")

    # ---------------------------------
    # 리밸런싱 알림
    # ---------------------------------
    if port_vol > 20:
        st.warning("변동성 높음 → 채권/금 확대 리밸런싱 권장")
    else:
        st.success("리밸런싱 필요 낮음")

    st.divider()

    # ---------------------------------
    # 고도화 리셋
    # ---------------------------------
    if st.button("전략 다시 계산"):
        st.session_state.picks, st.session_state.weights = generate_portfolio()
        st.cache_data.clear()
        st.rerun()
