import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
from datetime import datetime

# =====================================================
# 1. 설정 및 자산 유니버스
# =====================================================
st.set_page_config(layout="wide", page_title="AI Multi-Asset Manager")

# 자산군 정의 (미국/한국 혼합)
ASSET_DATABASE = {
    "US_RISK": ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA"],
    "KR_RISK": ["005930.KS", "000660.KS", "005380.KS", "035420.KS", "069500.KS"], # 삼성, 하이닉스, 현대차, 네이버, KODEX200
    "SAFE": ["TLT", "GLD", "IEF", "148070.KS"], # 미국채, 금, 한국10년국채(KODEX)
}
MARKET_BENCHMARK = "SPY"
FX_TICKER = "USDKRW=X" # 원/달러 환율

# =====================================================
# 2. 데이터 엔진
# =====================================================
@st.cache_data(ttl=3600)
def fetch_financial_data(tickers):
    try:
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            return data["Close"]
        return data
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

# =====================================================
# 3. 메인 로직
# =====================================================
st.title("🧠 AI 글로벌 자산운용사 (US & KR Edition)")

# 모든 티커 수집 및 데이터 로드
all_tickers = ASSET_DATABASE["US_RISK"] + ASSET_DATABASE["KR_RISK"] + ASSET_DATABASE["SAFE"] + [MARKET_BENCHMARK, FX_TICKER]
all_prices = fetch_financial_data(all_tickers)

if all_prices is not None:
    # 실시간 환율 정보
    current_fx = all_prices[FX_TICKER].iloc[-1]
    
    # 세션 상태 초기화
    if "picks" not in st.session_state:
        # 미국 4종목, 한국 3종목, 안전자산 2종목 무작위 선정
        st.session_state.picks = (random.sample(ASSET_DATABASE["US_RISK"], 3) + 
                                  random.sample(ASSET_DATABASE["KR_RISK"], 3) + 
                                  random.sample(ASSET_DATABASE["SAFE"], 2))
        st.session_state.base_weights = np.array([1.0 / 8.0] * 8)

    picks = st.session_state.picks
    prices = all_prices[picks].dropna()
    returns = prices.pct_change().dropna()

    # 시장 국면 분석 (S&P 500 기준)
    spy = all_prices[MARKET_BENCHMARK]
    vol_20d = spy.pct_change().tail(20).std() * np.sqrt(252) * 100
    regime = "안정" if vol_20d < 15 else ("중립" if vol_20d < 25 else "위기")
    
    # AI 비중 조정 로직
    ai_weights = st.session_state.base_weights.copy()
    if regime == "위기":
        for i, t in enumerate(picks):
            if ".KS" in t: ai_weights[i] *= 0.8 # 위기 시 신흥국(한국) 비중 축소
            if t in ASSET_DATABASE["SAFE"]: ai_weights[i] *= 1.5 # 안전자산 확대
    ai_weights /= ai_weights.sum()

    # 성과 지표 계산
    port_ret = returns.dot(ai_weights)
    cum_ret = (1 + port_ret).cumprod()
    cagr = (cum_ret.iloc[-1] ** (252 / len(returns)) - 1) * 100
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() * 100

    # --- UI 레이아웃 ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🌎 Global Portfolio Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("예상 연수익률", f"{cagr:.2f}%")
        m2.metric("최대 낙폭(MDD)", f"{mdd:.2f}%")
        m3.metric("현재 환율 (USD/KRW)", f"{current_fx:,.1f}원")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name="Portfolio", line=dict(color='#00FFAA', width=2)))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⚖️ Rebalancing")
        base_currency = st.radio("기준 통화 선택", ["USD ($)", "KRW (₩)"])
        total_inv = st.number_input("투자 원금", value=10000000 if base_currency == "KRW (₩)" else 10000)

        # 리밸런싱 테이블 구성
        rebalance_data = []
        latest_px = prices.iloc[-1]

        for i, t in enumerate(picks):
            # 목표 금액 계산 (통화 환산 반영)
            target_val_usd = (total_inv / current_fx if base_currency == "KRW (₩)" else total_inv) * ai_weights[i]
            
            # 종목별 현재가 (KR 종목은 원화, US 종목은 달러)
            price = latest_px[t]
            
            if ".KS" in t: # 한국 주식일 경우
                # USD 기준 목표액을 다시 KRW로 환산하여 수량 계산
                target_qty = (target_val_usd * current_fx) / price
                currency_unit = "KRW"
            else: # 미국 주식일 경우
                target_qty = target_val_usd / price
                currency_unit = "USD"

            rebalance_data.append({
                "Ticker": t,
                "Weight": f"{ai_weights[i]*100:.1f}%",
                "Qty": f"{int(target_qty)}주",
                "Currency": currency_unit
            })

        st.dataframe(pd.DataFrame(rebalance_data), use_container_width=True, hide_index=True)
        
        if st.button("🔄 유니버스 교체", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

st.divider()
st.caption("알림: 한국 주식 티커는 Yahoo Finance 기준 '.KS'(코스피) 또는 '.KQ'(코스닥) 접미사가 붙습니다. 환율 데이터는 실시간 USDKRW=X 티커를 참조합니다.")
