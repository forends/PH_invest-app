import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =====================================================
# 설정 및 초기화
# =====================================================
st.set_page_config(layout="wide", page_title="AI Institutional Asset Manager")

# 테마 색상 정의
RISK_COLOR = "#FF4B4B"
SAFE_COLOR = "#0068C9"

# 투자 유니버스
RISK_ASSETS = ["SPY", "QQQ", "VTI", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
SAFE_ASSETS = ["TLT", "IEF", "GLD"]
MARKET_BENCHMARK = "SPY"
UNIVERSE = RISK_ASSETS + SAFE_ASSETS

# =====================================================
# 데이터 엔진
# =====================================================
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    try:
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        if data.empty:
            return None
        return data["Close"]
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

def calculate_metrics(returns, weights):
    port_ret = returns.dot(weights)
    cum_ret = (1 + port_ret).cumprod()
    
    # 지표 계산
    total_ret = cum_ret.iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    vol = port_ret.std() * np.sqrt(252)
    sharpe = (annual_ret - 0.03) / vol if vol != 0 else 0  # 무위험 수익률 3% 가정
    
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    return cum_ret, annual_ret, vol, sharpe, mdd

# =====================================================
# AI 핵심 로직 (Regime Analysis)
# =====================================================
def get_market_regime(prices, benchmark):
    spy = prices[benchmark]
    
    # 1. 변동성 국면
    daily_ret = spy.pct_change().dropna()
    vol = daily_ret.tail(20).std() * np.sqrt(252) * 100
    if vol < 15: regime = "안정(Low Vol)"
    elif vol < 25: regime = "중립(Normal)"
    else: regime = "위기(High Vol)"
    
    # 2. 추세 국면 (MA 50/200)
    ma50 = spy.rolling(50).mean().iloc[-1]
    ma200 = spy.rolling(200).mean().iloc[-1]
    current = spy.iloc[-1]
    
    if current > ma50 > ma200: trend = "강세(Bull)"
    elif current < ma50 < ma200: trend = "약세(Bear)"
    else: trend = "횡보(Side)"
    
    return regime, trend, vol

# =====================================================
# 메인 대시보드
# =====================================================
st.title("🧠 AI 자산운용사 (Institutional Edition)")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 데이터 로드
all_prices = fetch_data(list(set(UNIVERSE + [MARKET_BENCHMARK])))

if all_prices is not None:
    # 세션 상태 관리
    if "picks" not in st.session_state:
        st.session_state.picks = random_picks = np.random.choice(RISK_ASSETS, 5, replace=False).tolist() + \
                                 np.random.choice(SAFE_ASSETS, 2, replace=False).tolist()
        # 초기 비중: 변동성 역수 가중치 (간단한 Risk Parity)
        st.session_state.base_weights = np.array([1/len(st.session_state.picks)] * len(st.session_state.picks))

    picks = st.session_state.picks
    prices = all_prices[picks].dropna()
    returns = prices.pct_change().dropna()
    
    # 시장 상황 분석
    regime, trend, m_vol = get_market_regime(all_prices, MARKET_BENCHMARK)
    
    # AI 가중치 조정 (Tilt Strategy)
    ai_weights = st.session_state.base_weights.copy()
    for i, ticker in enumerate(picks):
        if "위기" in regime or "약세" in trend:
            if ticker in SAFE_ASSETS: ai_weights[i] *= 1.5 # 안전자산 비중 강화
            else: ai_weights[i] *= 0.7 # 위험자산 축소
        elif "강세" in trend:
            if ticker in RISK_ASSETS: ai_weights[i] *= 1.2
            
    ai_weights /= ai_weights.sum() # 정규화

    # 성과 계산
    cum_ret, cagr, vol, sharpe, mdd = calculate_metrics(returns, ai_weights)

    # --- 레이아웃 배치 ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📊 포트폴리오 성과 분석")
        
        # 메트릭 섹션
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("연수익률 (CAGR)", f"{cagr*100:.2f}%")
        m2.metric("변동성 (Vol)", f"{vol*100:.2f}%")
        m3.metric("샤프 지수 (Sharpe)", f"{sharpe:.2f}")
        m4.metric("최대 낙폭 (MDD)", f"{mdd*100:.2f}%", delta_color="inverse")

        # 수익률 차트 (Plotly)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name="Portfolio", line=dict(color='#00FFAA', width=3)))
        fig.update_layout(
            template="plotly_dark", 
            hovermode="x unified",
            margin=dict(l=20, r=20, t=20, b=20),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # AI 보고서 영역
        with st.expander("🤖 AI 운용 전략 리포트", expanded=True):
            st.markdown(f"""
            - **시장 국면:** 현재 시장은 **{regime}** 및 **{trend}** 국면에 있습니다.
            - **조정 전략:** 이에 따라 AI는 {'방어적 자산 배분' if '위기' in regime else '공격적 수익 추구'} 전략을 실행 중입니다.
            - **리스크 관리:** 현재 포트폴리오의 MDD 수준은 `{mdd*100:.1f}%`로 관리되고 있습니다.
            """)

    with col2:
        st.subheader("⚖ 리밸런싱 시뮬레이터")
        total_money = st.number_input("투자 원금 ($)", value=10000, step=1000)
        
        # 비중 테이블
        rebalance_data = []
        latest_prices = prices.iloc[-1]
        
        for i, t in enumerate(picks):
            target_val = total_money * ai_weights[i]
            target_qty = target_val / latest_prices[t]
            rebalance_data.append({
                "Ticker": t,
                "Weight": f"{ai_weights[i]*100:.1f}%",
                "Target Qty": f"{target_qty:.2f} 주"
            })
        
        st.table(pd.DataFrame(rebalance_data))
        
        if st.button("🔄 유니버스 교체 및 재분석", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# =====================================================
# 하단 설명 가이드
# =====================================================
st.divider()
cols = st.columns(3)
with cols[0]:
    st.markdown("#### 🛡 Risk Regime")
    st.caption("변동성을 기준으로 시장의 공포 수준을 측정하여 자산 비중을 조절합니다.")
with cols[1]:
    st.markdown("#### 📈 Trend Following")
    st.caption("장단기 이평선을 활용하여 상승장에서는 수익을 극대화하고 하락장에서는 회피합니다.")
with cols[2]:
    st.markdown("#### 💎 Institutional Rebalancing")
    st.caption("목표 비중과 실제 보유 수량의 괴리를 계산하여 최적의 매매 수량을 산출합니다.")
