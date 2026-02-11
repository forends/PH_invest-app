import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("AI Portfolio Manager - BlackRock Style System")

# =====================================================
# 기본 세팅
# =====================================================
UNIVERSE = [
    "SPY","QQQ","VTI","IWM","VEA","VWO",
    "TLT","IEF","GLD",
    "AAPL","MSFT","NVDA","AMZN","GOOGL"
]

SAFE_ASSETS = ["TLT", "IEF", "GLD"]
MARKET = "SPY"
TARGET_VOL = 15  # 목표 변동성

# =====================================================
# 데이터
# =====================================================
@st.cache_data
def load_prices(tickers):
    df = yf.download(tickers, period="1y", auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            df = df["Close"]
        else:
            df = df.xs(df.columns.levels[0][0], axis=1, level=0)

    return df.dropna(how="all")

# =====================================================
# 포트폴리오 생성
# =====================================================
def generate_portfolio():
    picks = random.sample(UNIVERSE, 8)
    weights = np.random.dirichlet(np.ones(len(picks)), size=1)[0]
    return picks, weights


if "picks" not in st.session_state:
    st.session_state.picks, st.session_state.weights = generate_portfolio()

picks = st.session_state.picks
weights = st.session_state.weights

# =====================================================
# 가격
# =====================================================
prices = load_prices(picks + [MARKET])
latest_price = prices[picks].iloc[-1]

returns = prices.pct_change().dropna()
asset_returns = returns[picks]
market_returns = returns[MARKET]

# =====================================================
# 포트폴리오 수익률
# =====================================================
port_daily = asset_returns.dot(weights)
cum = (1 + port_daily).cumprod()

# =====================================================
# 변동성
# =====================================================
vol = float(port_daily.std() * np.sqrt(252) * 100)

# =====================================================
# 시장 레짐 판단
# =====================================================
market_vol = market_returns.std() * np.sqrt(252) * 100

if market_vol > 25:
    regime = "Crisis"
elif market_vol > 18:
    regime = "Risk Off"
elif market_vol < 12:
    regime = "Risk On"
else:
    regime = "Neutral"

# =====================================================
# 리스크 기여도 (Risk Contribution)
# =====================================================
cov_matrix = asset_returns.cov() * 252

portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))

marginal_contrib = np.dot(cov_matrix, weights) / portfolio_var
risk_contrib = weights * marginal_contrib
risk_contrib = risk_contrib * 100

risk_df = pd.DataFrame({
    "Ticker": picks,
    "Risk Contribution(%)": risk_contrib
})

# =====================================================
# 상관관계
# =====================================================
corr = asset_returns.corr()

# =====================================================
# 변동성 타겟팅
# =====================================================
scale = TARGET_VOL / vol if vol != 0 else 1

adj_weights = weights * scale
adj_weights = adj_weights / adj_weights.sum()

# =====================================================
# 위기 시 방어 확대
# =====================================================
if regime in ["Risk Off", "Crisis"]:
    for i, t in enumerate(picks):
        if t in SAFE_ASSETS:
            adj_weights[i] += 0.05

    adj_weights = adj_weights / adj_weights.sum()

# =====================================================
# AI 코멘트
# =====================================================
def ai_comment():
    if regime == "Crisis":
        return "위기 구간 → 생존 우선, 방어 자산 대폭 확대"
    if regime == "Risk Off":
        return "리스크 축소 필요 → 변동성 관리"
    if regime == "Risk On":
        return "위험자산 확대 가능"
    return "균형 운용 권장"

# =====================================================
# 레이아웃
# =====================================================
left, right = st.columns([3,1])

# =====================================================
# 좌측
# =====================================================
with left:
    st.subheader("Risk Architecture")

    k1, k2 = st.columns(2)
    k1.metric("Portfolio Vol", f"{vol:.2f}%")
    k2.metric("Market Regime", regime)

    st.line_chart(pd.DataFrame({"Portfolio": cum}), use_container_width=True)

    st.divider()

    st.subheader("Risk Contribution")
    st.dataframe(risk_df, use_container_width=True)

    st.divider()

    st.subheader("Correlation Matrix")
    st.dataframe(corr, use_container_width=True)

    st.divider()
    st.success(ai_comment())

# =====================================================
# 우측 : 실행 엔진
# =====================================================
with right:
    st.subheader("AI Execution Engine")

    total_money = st.number_input("총 자산 ($)", value=10000)

    current_shares = {}
    for t in picks:
        current_shares[t] = st.number_input(
            f"{t}", min_value=0, value=0, key=f"s_{t}"
        )

    current_values = {t: current_shares[t] * latest_price[t] for t in picks}

    rebalance = []

    for t, w in zip(picks, adj_weights):
        target = total_money * w
        diff = target - current_values[t]
        diff_share = int(diff // latest_price[t])

        if diff_share > 0:
            action = "매수"
        elif diff_share < 0:
            action = "매도"
        else:
            action = "유지"

        rebalance.append([t, round(w*100,2), diff_share, action])

    df = pd.DataFrame(
        rebalance,
        columns=["Ticker","AI목표비중","주문수량","액션"]
    )

    st.dataframe(df, use_container_width=True)

    st.divider()

    if st.button("새 포트폴리오 생성"):
        st.session_state.picks, st.session_state.weights = generate_portfolio()
        st.cache_data.clear()
        st.rerun()