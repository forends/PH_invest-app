import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("🧠 AI 자산운용사 (Institutional Level)")

# =====================================================
# 투자 유니버스
# =====================================================
RISK = [
    "SPY","QQQ","VTI","IWM",
    "AAPL","MSFT","NVDA","AMZN","GOOGL"
]

SAFE = ["TLT","IEF","GLD"]

MARKET = "SPY"

UNIVERSE = RISK + SAFE

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
# 포트폴리오 초기 생성
# =====================================================
def generate_portfolio():
    picks = random.sample(UNIVERSE, 8)
    weights = np.random.dirichlet(np.ones(len(picks)), size=1)[0]
    return picks, weights


if "picks" not in st.session_state:
    st.session_state.picks, st.session_state.weights = generate_portfolio()

picks = st.session_state.picks
base_weights = np.array(st.session_state.weights)

prices = load_price(list(set(picks + [MARKET])))
latest_price = prices[picks].iloc[-1]

# =====================================================
# 수익률
# =====================================================
returns = prices[picks].pct_change().dropna()
market_ret = prices[MARKET].pct_change().dropna()

# =====================================================
# 시장 리스크 국면 판단
# =====================================================
market_vol = market_ret.std() * np.sqrt(252) * 100

if market_vol < 15:
    regime = "위험선호"
elif market_vol < 25:
    regime = "중립"
else:
    regime = "위기"

# =====================================================
# 추세 판단 (50 / 200 MA)
# =====================================================
spy = prices[MARKET]
ma50 = spy.rolling(50).mean().iloc[-1]
ma200 = spy.rolling(200).mean().iloc[-1]
now = spy.iloc[-1]

if now > ma50 > ma200:
    trend = "강한상승"
elif now > ma200:
    trend = "상승"
elif now < ma50 < ma200:
    trend = "하락"
else:
    trend = "중립"

# =====================================================
# AI 비중 조정
# =====================================================
ai_weights = base_weights.copy()

for i, t in enumerate(picks):

    # 위기 or 하락 → 안전자산 확대
    if regime == "위기" or trend == "하락":
        if t in SAFE:
            ai_weights[i] += 0.05
        else:
            ai_weights[i] -= 0.03

    # 강한 상승 → 위험자산 확대
    elif regime == "위험선호" and trend == "강한상승":
        if t in RISK:
            ai_weights[i] += 0.03

# 음수 제거 + 재정규화
ai_weights = np.clip(ai_weights, 0, None)
ai_weights = ai_weights / ai_weights.sum()

# =====================================================
# 포트폴리오 성과
# =====================================================
port_daily = returns.dot(ai_weights)
cum = (1 + port_daily).cumprod()

days = len(cum)
cagr = (cum.iloc[-1] ** (252/days) - 1) * 100
vol = port_daily.std() * np.sqrt(252) * 100

rf = 0.02
sharpe = (port_daily.mean()*252 - rf) / (port_daily.std()*np.sqrt(252))

rolling_max = cum.cummax()
drawdown = cum / rolling_max - 1
mdd = drawdown.min() * 100

# =====================================================
# AI 운용 보고서
# =====================================================
def ai_report():
    text = f"현재 시장 변동성은 {market_vol:.1f}%로 '{regime}' 국면입니다. "
    text += f"추세는 '{trend}' 상태입니다. "

    if trend == "하락":
        text += "하락 추세 감지 → 방어 자산을 확대합니다. "
    elif trend == "강한상승":
        text += "강한 상승 추세 → 위험 자산 비중을 확대합니다. "
    else:
        text += "균형 포지션을 유지합니다. "

    text += f"예상 포트폴리오 변동성은 {vol:.1f}% 수준입니다."

    return text


# =====================================================
# 레이아웃
# =====================================================
left, right = st.columns([3,1])

# =====================================================
# 대시보드
# =====================================================
with left:
    st.subheader("📈 Performance Dashboard")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CAGR", f"{cagr:.2f}%")
    k2.metric("Volatility", f"{vol:.2f}%")
    k3.metric("Sharpe", f"{sharpe:.2f}")
    k4.metric("MDD", f"{mdd:.2f}%")

    st.line_chart(cum)

    st.info(ai_report())

# =====================================================
# 리밸런싱
# =====================================================
with right:
    st.subheader("⚖ 리밸런싱")

    total_money = st.number_input("총 자산 ($)", value=10000)

    st.write("### 현재 보유 수량")

    current_shares = {}
    for t in picks:
        current_shares[t] = st.number_input(f"{t}", min_value=0, value=0)

    current_values = {t: current_shares[t] * latest_price[t] for t in picks}
    current_total = sum(current_values.values())

    if current_total == 0:
        st.info("수량 입력 시 계산됩니다.")
    else:
        rebalance = []

        for i, t in enumerate(picks):
            target_value = total_money * ai_weights[i]
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
                round(ai_weights[i]*100,2),
                current_shares[t],
                diff_shares,
                action
            ])

        df = pd.DataFrame(
            rebalance,
            columns=["Ticker","AI 목표비중(%)","현재수량","변경수량","액션"]
        )

        st.dataframe(df, use_container_width=True)

# =====================================================
# 용어 설명
# =====================================================
st.divider()
st.subheader("📘 용어 설명")
st.caption("CAGR → 연평균 복리 수익률")
st.caption("Volatility → 가격 변동 위험")
st.caption("Sharpe → 위험 대비 효율")
st.caption("MDD → 최대 손실폭")
st.caption("리밸런싱 → 목표 비율로 맞추는 매매")

# =====================================================
# 재시작
# =====================================================
if st.button("🔄 새 포트폴리오 생성"):
    st.session_state.picks, st.session_state.weights = generate_portfolio()
    st.cache_data.clear()
    st.rerun()