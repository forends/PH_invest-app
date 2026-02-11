import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("AI 자산운용사 시스템")

# =====================================================
# 설정
# =====================================================
UNIVERSE = [
    "SPY","QQQ","VTI","IWM","VEA","VWO",
    "TLT","IEF","GLD",
    "AAPL","MSFT","NVDA","AMZN","GOOGL"
]

SAFE = ["TLT", "IEF", "GLD"]
MARKET = "SPY"
TARGET_VOL = 15

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
market_vol = float(market_returns.std() * np.sqrt(252) * 100)

# =====================================================
# 시장 국면 판단
# =====================================================
if market_vol > 25:
    regime = "위기"
elif market_vol > 18:
    regime = "위험회피"
elif market_vol < 12:
    regime = "위험선호"
else:
    regime = "중립"

# =====================================================
# 변동성 타겟 비중 조정
# =====================================================
scale = TARGET_VOL / vol if vol != 0 else 1
ai_weights = weights * scale
ai_weights = ai_weights / ai_weights.sum()

# 위기 시 안전자산 추가 확대
if regime in ["위기", "위험회피"]:
    for i, t in enumerate(picks):
        if t in SAFE:
            ai_weights[i] += 0.05
    ai_weights = ai_weights / ai_weights.sum()

# =====================================================
# 리스크 기여도
# =====================================================
cov = asset_returns.cov() * 252
port_var = np.dot(ai_weights.T, np.dot(cov, ai_weights))
marginal = np.dot(cov, ai_weights) / port_var
risk_contrib = ai_weights * marginal * 100

# =====================================================
# AI 운용 설명 생성
# =====================================================
def ai_report():
    text = f"현재 시장 변동성은 {market_vol:.1f}% 수준으로 '{regime}' 국면으로 판단됩니다. "
    
    if regime == "위기":
        text += "대규모 손실 가능성을 줄이기 위해 채권과 금 비중을 확대했습니다. "
    elif regime == "위험회피":
        text += "주식 비중을 일부 줄이고 방어 자산을 늘리는 전략을 사용합니다. "
    elif regime == "위험선호":
        text += "시장 환경이 안정적이므로 성장 자산 비중을 확대할 수 있습니다. "
    else:
        text += "균형 잡힌 자산 배분을 유지합니다. "

    text += f"현재 포트폴리오 변동성은 {vol:.1f}% 입니다."
    return text


# =====================================================
# 레이아웃
# =====================================================
left, right = st.columns([3,1])

# =====================================================
# 좌측 : 운용 본부
# =====================================================
with left:
    st.subheader("📊 포트폴리오 현황")

    k1, k2 = st.columns(2)
    k1.metric("포트폴리오 변동성", f"{vol:.2f}%")
    k2.metric("시장 국면", regime)

    st.line_chart(pd.DataFrame({"포트폴리오": cum}), use_container_width=True)

    st.divider()

    st.subheader("📉 자산별 리스크 기여도")
    risk_df = pd.DataFrame({
        "종목": picks,
        "리스크기여도(%)": risk_contrib
    })
    st.dataframe(risk_df, use_container_width=True)

    st.divider()

    st.subheader("🧠 AI 운용 판단 리포트")
    st.info(ai_report())

# =====================================================
# 우측 : 매매 실행
# =====================================================
with right:
    st.subheader("💰 매매 계산기")

    total_money = st.number_input("총 투자 금액 ($)", value=10000)

    st.write("현재 보유 수량 입력")
    current_shares = {}
    for t in picks:
        current_shares[t] = st.number_input(
            f"{t}", min_value=0, value=0, key=f"hold_{t}"
        )

    current_values = {t: current_shares[t] * latest_price[t] for t in picks}

    rebalance = []
    for t, w in zip(picks, ai_weights):
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
        columns=["종목","AI목표비중","주문수량","액션"]
    )
    st.dataframe(df, use_container_width=True)

    st.divider()

    if st.button("🔄 새로운 전략 받기"):
        st.session_state.picks, st.session_state.weights = generate_portfolio()
        st.cache_data.clear()
        st.rerun()