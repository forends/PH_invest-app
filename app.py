import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("AI Portfolio Manager - Pro")

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

    # 멀티 인덱스 대응
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
    picks = random.sample(STOCK_UNIVERSE, 8)
    weights = np.random.dirichlet(np.ones(len(picks)), size=1)[0]
    return picks, weights

if "picks" not in st.session_state:
    st.session_state.picks, st.session_state.weights = generate_portfolio()

picks = st.session_state.picks
weights = st.session_state.weights

# =====================================================
# 가격
# =====================================================
prices = load_price(picks)

if prices.empty:
    st.error("가격 데이터를 불러올 수 없습니다.")
    st.stop()

latest_price = prices.iloc[-1]

# =====================================================
# 수익률
# =====================================================
returns = prices.pct_change().dropna()

if returns.empty:
    st.error("수익률 계산 불가")
    st.stop()

port_daily = returns.dot(weights)
cum = (1 + port_daily).cumprod()

# =====================================================
# 📈 성과 지표
# =====================================================
days = len(cum)

cagr = float((cum.iloc[-1] ** (252/days) - 1) * 100)
vol = float(port_daily.std() * np.sqrt(252) * 100)

rf = 0.02
if port_daily.std() == 0:
    sharpe = 0.0
else:
    sharpe = float((port_daily.mean()*252 - rf) / (port_daily.std()*np.sqrt(252)))

rolling_max = cum.cummax()
drawdown = cum / rolling_max - 1
mdd = float(drawdown.min() * 100)

# =====================================================
# AI 의사결정
# =====================================================
def ai_decision(cagr, vol, sharpe, mdd):
    if sharpe > 1 and mdd > -15:
        return "전략 우수 → 유지 또는 확대 가능"
    elif vol > 25:
        return "변동성 높음 → 방어 자산 확대 권장"
    elif mdd < -25:
        return "낙폭 큼 → 일부 비중 축소 검토"
    else:
        return "중립 → 정기 리밸런싱 유지"

decision = ai_decision(cagr, vol, sharpe, mdd)

# =====================================================
# 레이아웃
# =====================================================
left, right = st.columns([3,1])

# =====================================================
# 좌측 : 대시보드
# =====================================================
with left:
    st.subheader("Performance Dashboard")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CAGR", f"{cagr:.2f}%")
    k2.metric("Volatility", f"{vol:.2f}%")
    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    k4.metric("MDD", f"{mdd:.2f}%")

    chart_df = pd.DataFrame({"Portfolio": cum})
    st.line_chart(chart_df, use_container_width=True)

    st.success(decision)

# =====================================================
# 우측 : 리밸런싱
# =====================================================
with right:
    st.subheader("Rebalancing")

    total_money = st.number_input("총 자산 ($)", value=10000)

    st.write("### 현재 보유 수량")

    current_shares = {}
    for t in picks:
        current_shares[t] = st.number_input(f"{t}", min_value=0, value=0, key=f"share_{t}")

    current_values = {t: current_shares[t] * latest_price[t] for t in picks}
    current_total = sum(current_values.values())

    if current_total == 0:
        st.info("수량 입력 시 계산됩니다.")
    else:
        rebalance = []

        for t, w in zip(picks, weights):
            target_value = total_money * w
            diff_value = target_value - current_values[t]
            diff_shares = int(diff_value // latest_price[t])

            if diff_shares > 0:
                action = "매수"
            elif diff_shares < 0:
                action = "매도"
            else:
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
    st.caption("CAGR → 연평균 복리 수익률")
    st.caption("Volatility → 변동성, 위험도 지표")
    st.caption("Sharpe Ratio → 위험 대비 수익 효율")
    st.caption("MDD → 최대 손실 구간")
    st.caption("리밸런싱 → 목표 비율로 되돌리는 매매")

    st.divider()

    if st.button("AI 전략 다시 계산"):
        st.session_state.picks, st.session_state.weights = generate_portfolio()
        st.cache_data.clear()
        st.rerun()