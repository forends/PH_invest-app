import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("AI Portfolio Manager - Quant Fund Edition")

# =====================================================
# 기본 설정
# =====================================================
UNIVERSE = [
    "SPY","QQQ","VTI","IWM","VEA","VWO",
    "TLT","IEF","GLD",
    "AAPL","MSFT","NVDA","AMZN","GOOGL"
]

MARKET = "SPY"
RISK_FREE = 0.02

# 섹터 분류 (간단 버전)
SECTOR_MAP = {
    "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","GOOGL":"Tech","AMZN":"Consumer",
    "SPY":"Index","QQQ":"Tech","VTI":"Index","IWM":"SmallCap",
    "VEA":"International","VWO":"Emerging",
    "TLT":"Bond","IEF":"Bond","GLD":"Gold"
}

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

port_daily = asset_returns.dot(weights)
cum = (1 + port_daily).cumprod()

# =====================================================
# 성과
# =====================================================
days = len(cum)
cagr = float((cum.iloc[-1] ** (252/days) - 1) * 100)
vol = float(port_daily.std() * np.sqrt(252) * 100)

if port_daily.std() == 0:
    sharpe = 0.0
else:
    sharpe = float((port_daily.mean()*252 - RISK_FREE) /
                   (port_daily.std()*np.sqrt(252)))

rolling_max = cum.cummax()
drawdown = cum / rolling_max - 1
mdd = float(drawdown.min() * 100)

# =====================================================
# 시장 레짐 판단
# =====================================================
market_vol = market_returns.std() * np.sqrt(252) * 100

if market_vol > 25:
    regime = "Risk OFF"
elif market_vol < 15:
    regime = "Risk ON"
else:
    regime = "Neutral"

# =====================================================
# 섹터 노출
# =====================================================
sector_weight = {}
for t, w in zip(picks, weights):
    sec = SECTOR_MAP.get(t, "Other")
    sector_weight[sec] = sector_weight.get(sec, 0) + w

sector_df = pd.DataFrame(
    {"Sector": sector_weight.keys(),
     "Weight": [v*100 for v in sector_weight.values()]}
)

# =====================================================
# 모멘텀 팩터
# =====================================================
momentum = asset_returns.mean() * 252
mom_score = float(momentum.mean() * 100)

# =====================================================
# 위험 점수
# =====================================================
risk_score = (vol * 0.4) + (abs(mdd) * 0.3) + (market_vol * 0.3)

# =====================================================
# AI 판단
# =====================================================
def ai_comment():
    msg = f"시장 국면: {regime} / "
    if regime == "Risk OFF":
        msg += "방어자산 확대 권장."
    elif regime == "Risk ON":
        msg += "공격적 운용 가능."
    else:
        msg += "균형 유지 필요."
    return msg

# =====================================================
# 레이아웃
# =====================================================
left, right = st.columns([3,1])

# =====================================================
# 좌측
# =====================================================
with left:
    st.subheader("Quant Dashboard")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CAGR", f"{cagr:.2f}%")
    k2.metric("Vol", f"{vol:.2f}%")
    k3.metric("Sharpe", f"{sharpe:.2f}")
    k4.metric("MDD", f"{mdd:.2f}%")

    k5, k6 = st.columns(2)
    k5.metric("Market Regime", regime)
    k6.metric("Momentum", f"{mom_score:.2f}%")

    st.line_chart(pd.DataFrame({"Portfolio": cum}), use_container_width=True)

    st.divider()

    st.subheader("Sector Exposure")
    st.dataframe(sector_df, use_container_width=True)

    st.divider()
    st.warning(f"Risk Score: {risk_score:.1f}")
    st.success(ai_comment())

# =====================================================
# 우측 : 매매
# =====================================================
with right:
    st.subheader("Execution")

    total_money = st.number_input("총 자산 ($)", value=10000)

    current_shares = {}
    for t in picks:
        current_shares[t] = st.number_input(
            f"{t}", min_value=0, value=0, key=f"shares_{t}"
        )

    current_values = {t: current_shares[t] * latest_price[t] for t in picks}
    current_total = sum(current_values.values())

    if current_total != 0:
        rebalance = []
        for t, w in zip(picks, weights):
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
            columns=["Ticker","목표비중","주문수량","액션"]
        )
        st.dataframe(df, use_container_width=True)

    st.divider()

    if st.button("AI 새 전략 생성"):
        st.session_state.picks, st.session_state.weights = generate_portfolio()
        st.cache_data.clear()
        st.rerun()