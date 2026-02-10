import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# =====================================================
# Core-Satellite 포트폴리오 + 추천 이유
# =====================================================
PORT_INFO = {
    # Core
    "SPY": {"weight": 0.10, "reason": "미국 대형주 시장 대표"},
    "QQQ": {"weight": 0.15, "reason": "기술주 성장 엔진"},
    "VTI": {"weight": 0.10, "reason": "미국 전체 시장 분산"},

    # Growth
    "TQQQ": {"weight": 0.15, "reason": "상승장에서 수익 극대화"},
    "UPRO": {"weight": 0.10, "reason": "S&P500 레버리지"},
    "TECL": {"weight": 0.10, "reason": "빅테크 집중 레버리지"},

    # Theme
    "SMH": {"weight": 0.10, "reason": "반도체 슈퍼사이클"},
    "BOTZ": {"weight": 0.05, "reason": "AI/로봇 성장"},
    "SKYY": {"weight": 0.05, "reason": "클라우드 확대"},

    # Defense
    "SCHD": {"weight": 0.07, "reason": "배당 + 가치주"},
    "TLT": {"weight": 0.03, "reason": "위기 시 채권 헤지"}
}

TICKERS = list(PORT_INFO.keys())

# =====================================================
# 데이터 로드 (완전 안전 버전)
# =====================================================
@st.cache_data(ttl=3600)
def load_data(tickers):
    try:
        data = yf.download(tickers, period="1y", auto_adjust=True)

        # 단일 종목일 때와 복수 종목일 때 구조 통일
        if isinstance(data, pd.DataFrame):
            return data
        else:
            return pd.DataFrame(data)

    except Exception as e:
        st.error("데이터 다운로드 실패")
        return pd.DataFrame()

prices = load_data(TICKERS)

if prices.empty:
    st.stop()

# =====================================================
# 수익률 계산
# =====================================================
returns = prices.pct_change().dropna()

exp_returns = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)

# =====================================================
# 목표 수익률
# =====================================================
target = st.sidebar.slider("🎯 목표 연 수익률 (%)", 5, 60, 25)

# =====================================================
# 타이틀
# =====================================================
st.title("📊 AI 포트폴리오 전략 대시보드")
st.caption("Core-Satellite 기반 공격형 운용 모델")

# =====================================================
# 좌 / 우 분할
# =====================================================
left, right = st.columns([2, 1])

# =====================================================
# 왼쪽 : 종목 상세
# =====================================================
with left:
    st.header("🚀 추천 포트폴리오")

    total_return = 0
    total_vol = 0

    for t in TICKERS:
        if t not in returns.columns:
            continue

        w = PORT_INFO[t]["weight"]
        er = exp_returns[t] * 100
        vol = volatility[t] * 100

        total_return += er * w
        total_vol += vol * w

        # 위험도 표시
        if vol < 20:
            risk = "🟢 낮음"
        elif vol < 35:
            risk = "🟡 보통"
        else:
            risk = "🔴 높음"

        st.subheader(f"{t}")
        st.write(PORT_INFO[t]["reason"])
        st.write(f"비중: **{w*100:.0f}%**")
        st.write(f"예상 수익률: **{er:.1f}%**")
        st.write(f"위험도: {risk} ({vol:.1f}%)")
        st.divider()

    st.subheader("📈 포트폴리오 기대 수익률")
    st.write(f"### 👉 {total_return:.1f}%")

    st.subheader("⚠ 포트폴리오 변동성")
    st.write(f"### 👉 {total_vol:.1f}%")

    if total_return >= target:
        st.success("🎉 목표 수익률 달성 기대!")
    else:
        st.warning("목표 수익률에 부족 → 공격 자산 확대 검토")

# =====================================================
# 오른쪽 : 리밸런싱
# =====================================================
with right:
    st.header("🔄 리밸런싱 체크")

    weights = np.array([PORT_INFO[t]["weight"] for t in TICKERS])
    drift = np.abs(weights - weights.mean())

    if drift.max() > 0.08:
        st.warning("비중 편차 발생 → 리밸런싱 필요")
    else:
        st.success("현재 비중 안정")

    st.divider()

    st.header("📊 구성 비율")
    chart = pd.DataFrame({"ticker": TICKERS, "weight": weights})
    st.bar_chart(chart.set_index("ticker"))

# =====================================================
# 누적 수익률
# =====================================================
st.header("📈 최근 1년 누적 수익률")

cum = (1 + returns).cumprod()
st.line_chart(cum)
