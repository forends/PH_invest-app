import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("Professional Portfolio System - Rebalancing")

# =====================================================
# 투자 유니버스
# =====================================================
STOCK_UNIVERSE = [
    "SPY","QQQ","VTI","IWM","VEA","VWO",
    "TLT","IEF","GLD",
    "AAPL","MSFT","NVDA","AMZN","GOOGL"
]

# =====================================================
# 데이터
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
latest_price = prices.iloc[-1]

# =====================================================
# 수익률
# =====================================================
returns = prices.pct_change().dropna()
port_daily = returns.dot(weights)
cum = (1 + port_daily).cumprod()

# =====================================================
# 레이아웃
# =====================================================
left, right = st.columns([3,1])

# =====================================================
# 좌측 : 성과
# =====================================================
with left:
    st.subheader("Performance")

    exp_return = port_daily.mean()*252*100
    vol = port_daily.std()*np.sqrt(252)*100

    k1, k2 = st.columns(2)
    k1.metric("Expected Return", f"{exp_return:.2f}%")
    k2.metric("Volatility", f"{vol:.2f}%")

    st.line_chart(cum)

# =====================================================
# 우측 : 리밸런싱 엔진
# =====================================================
with right:
    st.subheader("Rebalancing Engine")

    total_money = st.number_input("총 자산 ($)", value=10000)

    st.write("### 현재 보유 수량 입력")

    current_shares = {}
    for t in picks:
        current_shares[t] = st.number_input(f"{t}", min_value=0, value=0)

    # 현재 평가 금액
    current_values = {t: current_shares[t] * latest_price[t] for t in picks}
    current_total = sum(current_values.values())

    if current_total == 0:
        st.info("보유 수량을 입력하면 리밸런싱 계산 시작")
    else:
        rebalance = []

        for t, w in zip(picks, weights):
            target_value = total_money * w
            diff_value = target_value - current_values[t]
            diff_shares = int(diff_value // latest_price[t])

            action = "매수" if diff_shares > 0 else "매도"
            if diff_shares == 0:
                action = "유지"

            rebalance.append([
                t,
                round(w*100,2),
                current_shares[t],
                diff_shares,
                action
            ])

        df = pd.DataFrame(
            rebalance,
            columns=["Ticker","목표비중(%)","현재수량","변경수량","액션"]
        )

        st.dataframe(df, use_container_width=True)

    st.divider()

    # =====================================================
    # 용어 해설
    # =====================================================
    st.subheader("용어 설명")

    st.caption("Expected Return → 과거 데이터를 기준으로 예상되는 연간 수익률")
    st.caption("Volatility → 가격 변동성, 높을수록 위험")
    st.caption("목표비중 → AI가 추천하는 이상적인 투자 비율")
    st.caption("변경수량 → 목표비중에 맞추기 위해 사고 팔아야 할 주식 수")
    st.caption("리밸런싱 → 비율이 틀어졌을 때 다시 맞추는 작업")

    st.divider()

    if st.button("AI 전략 다시 계산"):
        st.session_state.picks, st.session_state.weights = generate_portfolio()
        st.cache_data.clear()
        st.rerun()