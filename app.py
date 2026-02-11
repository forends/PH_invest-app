import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("AI Portfolio Manager - Institutional Edition")

# =====================================================
# 투자 유니버스
# =====================================================
UNIVERSE = [
    "SPY","QQQ","VTI","IWM","VEA","VWO",
    "TLT","IEF","GLD",
    "AAPL","MSFT","NVDA","AMZN","GOOGL"
]

MARKET = "SPY"
RISK_FREE = 0.02

# =====================================================
# 데이터 로드
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
# 가격 데이터
# =====================================================
tickers = picks + [MARKET]
prices = load_prices(tickers)

if prices.empty:
    st.error("가격 데이터 로딩 실패")
    st.stop()

latest_price = prices[picks].iloc[-1]

# =====================================================
# 수익률
# =====================================================
returns = prices.pct_change().dropna()
asset_returns = returns[picks]
market_returns = returns[MARKET]

if asset_returns.empty:
    st.error("수익률 계산 실패")
    st.stop()

port_daily = asset_returns.dot(weights)
cum = (1 + port_daily).cumprod()

# =====================================================
# 핵심 성과 지표
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
# 베타 / 초과 수익
# =====================================================
cov = np.cov(port_daily, market_returns)[0][1]
market_var = np.var(market_returns)
beta = float(cov / market_var) if market_var != 0 else 0

alpha = float((port_daily.mean() - market_returns.mean()) * 252 * 100)

# =====================================================
# VaR (95%)
# =====================================================
var_95 = float(np.percentile(port_daily, 5) * 100)

# =====================================================
# 종목 기여도
# =====================================================
contribution = (asset_returns.mean() * weights) * 252 * 100

# =====================================================
# AI 리밸런싱 판단
# =====================================================
def ai_decision(sharpe, mdd, vol, beta):
    if sharpe > 1.2 and mdd > -15:
        return "전략 매우 양호 → 공격 유지 가능"
    if beta > 1.2:
        return "시장 민감도 높음 → 변동성 주의"
    if mdd < -25:
        return "낙폭 과다 → 비중 축소 검토"
    if vol > 30:
        return "리스크 확대 구간 → 일부 방어 필요"
    return "정상 범위 → 정기 리밸런싱"

decision = ai_decision(sharpe, mdd, vol, beta)

# =====================================================
# 레이아웃
# =====================================================
left, right = st.columns([3,1])

# =====================================================
# 좌측 : 기관급 대시보드
# =====================================================
with left:
    st.subheader("Portfolio Performance")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CAGR", f"{cagr:.2f}%")
    k2.metric("Volatility", f"{vol:.2f}%")
    k3.metric("Sharpe", f"{sharpe:.2f}")
    k4.metric("MDD", f"{mdd:.2f}%")

    k5, k6, k7 = st.columns(3)
    k5.metric("Beta", f"{beta:.2f}")
    k6.metric("Alpha", f"{alpha:.2f}%")
    k7.metric("VaR 95%", f"{var_95:.2f}%")

    chart_df = pd.DataFrame({"Portfolio": cum})
    st.line_chart(chart_df, use_container_width=True)

    st.success(decision)

    st.divider()

    st.subheader("종목 수익 기여도 (연율)")
    contrib_df = pd.DataFrame({
        "Ticker": picks,
        "Contribution(%)": contribution.values
    })
    st.dataframe(contrib_df, use_container_width=True)


# =====================================================
# 우측 : 실제 매매 엔진
# =====================================================
with right:
    st.subheader("Trade Engine")

    total_money = st.number_input("총 투자 자산 ($)", value=10000)

    st.write("### 현재 보유 수량")

    current_shares = {}
    for t in picks:
        current_shares[t] = st.number_input(
            f"{t}", min_value=0, value=0, key=f"shares_{t}"
        )

    current_values = {t: current_shares[t] * latest_price[t] for t in picks}
    current_total = sum(current_values.values())

    if current_total == 0:
        st.info("보유 수량 입력 시 계산")
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
            columns=["Ticker","목표비중","현재수량","변경수량","액션"]
        )

        st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("용어 해설")
    st.caption("CAGR → 연평균 복리 수익률")
    st.caption("Volatility → 수익률 흔들림 크기")
    st.caption("Sharpe → 위험 대비 성과")
    st.caption("MDD → 최대 손실 폭")
    st.caption("Beta → 시장 대비 민감도")
    st.caption("Alpha → 시장 초과 수익")
    st.caption("VaR → 최악 손실 예상치")

    st.divider()

    if st.button("AI 포트폴리오 재생성"):
        st.session_state.picks, st.session_state.weights = generate_portfolio()
        st.cache_data.clear()
        st.rerun()