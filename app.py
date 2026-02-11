import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random
from datetime import datetime

# =====================================================
# 1. 자산 정보 및 섹터 매핑
# =====================================================
st.set_page_config(layout="wide", page_title="AI Multi-Asset Manager Pro")

# 티커 정보 (회사명, 섹터)
TICKER_INFO = {
    "SPY": {"name": "S&P 500 지수", "sector": "지수(ETF)"},
    "QQQ": {"name": "나스닥 100", "sector": "지수(ETF)"},
    "NVDA": {"name": "엔비디아", "sector": "반도체/AI"},
    "AAPL": {"name": "애플", "sector": "빅테크"},
    "MSFT": {"name": "마이크로소프트", "sector": "빅테크"},
    "TSLA": {"name": "테슬라", "sector": "자동차/EV"},
    "005930.KS": {"name": "삼성전자", "sector": "반도체/AI"},
    "000660.KS": {"name": "SK하이닉스", "sector": "반도체/AI"},
    "005380.KS": {"name": "현대차", "sector": "자동차/EV"},
    "035420.KS": {"name": "NAVER", "sector": "플랫폼"},
    "069500.KS": {"name": "KODEX 200", "sector": "지수(ETF)"},
    "TLT": {"name": "미국 20년 국채", "sector": "채권"},
    "IEF": {"name": "미국 7-10년 국채", "sector": "채권"},
    "GLD": {"name": "금 현물", "sector": "원자재"},
    "148070.KS": {"name": "KODEX 10년 국채", "sector": "채권"},
    "USDKRW=X": {"name": "원/달러 환율", "sector": "외환"}
}

ASSET_DATABASE = {
    "US_RISK": ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA"],
    "KR_RISK": ["005930.KS", "000660.KS", "005380.KS", "035420.KS", "069500.KS"],
    "SAFE": ["TLT", "GLD", "IEF", "148070.KS"],
}
MARKET_BENCHMARK = "SPY"
FX_TICKER = "USDKRW=X"

# =====================================================
# 2. 데이터 엔진
# =====================================================
@st.cache_data(ttl=3600)
def fetch_data(tickers):
    try:
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        return data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

# =====================================================
# 3. 메인 애플리케이션
# =====================================================
st.title("🧠 AI 글로벌 자산운용사 (Pro Edition)")

all_tickers = list(TICKER_INFO.keys())
all_prices = fetch_data(all_tickers)

if all_prices is not None:
    current_fx = all_prices[FX_TICKER].iloc[-1]
    
    # 3-1. 포트폴리오 구성
    if "picks" not in st.session_state:
        st.session_state.picks = (random.sample(ASSET_DATABASE["US_RISK"], 3) + 
                                  random.sample(ASSET_DATABASE["KR_RISK"], 2) + 
                                  random.sample(ASSET_DATABASE["SAFE"], 2))
        st.session_state.base_weights = np.array([1.0 / len(st.session_state.picks)] * len(st.session_state.picks))

    picks = st.session_state.picks
    prices = all_prices[picks].dropna()
    returns = prices.pct_change().dropna()

    # 시장 상태 분석
    spy = all_prices[MARKET_BENCHMARK]
    vol_20d = spy.pct_change().tail(20).std() * np.sqrt(252) * 100
    regime = "안정" if vol_20d < 15 else ("중립" if vol_20d < 25 else "위기")
    
    # AI 비중 조정
    ai_weights = st.session_state.base_weights.copy()
    if regime == "위기":
        for i, t in enumerate(picks):
            if ".KS" in t: ai_weights[i] *= 0.7 
            if TICKER_INFO[t]['sector'] == "채권": ai_weights[i] *= 1.5
    ai_weights /= ai_weights.sum()

    # 성과 계산
    port_ret = returns.dot(ai_weights)
    cum_ret = (1 + port_ret).cumprod()
    cagr = (cum_ret.iloc[-1] ** (252 / len(returns)) - 1) * 100
    mdd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() * 100

    # =====================================================
    # 4. UI 레이아웃 (Tabs 적용)
    # =====================================================
    main_tab, risk_tab, rebalance_tab = st.tabs(["📈 성과 분석", "🔍 심층 리스크 분석", "⚖️ 리밸런싱 지시서"])

    # --- Tab 1: 성과 분석 ---
    with main_tab:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("포트폴리오 수익률 추이")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name="AI Portfolio", line=dict(color='#00FFAA', width=3)))
            fig_line.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_line, use_container_width=True)
        with col2:
            st.subheader("Key Metrics")
            st.metric("연평균 수익률(CAGR)", f"{cagr:.2f}%")
            st.metric("최대 낙폭(MDD)", f"{mdd:.2f}%")
            st.metric("시장 변동성", f"{vol_20d:.1f}%", delta=regime)
            st.info(f"봇 의견: 현재는 **{regime}** 국면으로 자산 배분을 조정했습니다.")

    # --- Tab 2: 심층 리스크 분석 (추가된 기능) ---
    with risk_tab:
        r_col1, r_col2 = st.columns(2)
        
        with r_col1:
            st.subheader("🏢 섹터별 배분 비중")
            sector_data = pd.DataFrame({
                "Sector": [TICKER_INFO[t]['sector'] for t in picks],
                "Weight": ai_weights
            }).groupby("Sector").sum().reset_index()
            
            fig_pie = px.pie(sector_data, values='Weight', names='Sector', 
                             hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with r_col2:
            st.subheader("🔗 자산 간 상관관계")
            corr_matrix = returns.corr()
            # 티커 대신 이름으로 표시
            corr_matrix.columns = [TICKER_INFO[t]['name'] for t in corr_matrix.columns]
            corr_matrix.index = [TICKER_INFO[t]['name'] for t in corr_matrix.index]
            
            fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                 color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_heat.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)

    # --- Tab 3: 리밸런싱 ---
    with rebalance_tab:
        st.subheader("매매 지시서")
        base_currency = st.radio("기준 통화", ["USD ($)", "KRW (₩)"], horizontal=True)
        total_inv = st.number_input("총 투자금액", value=10000000 if base_currency == "KRW (₩)" else 10000)

        reb_list = []
        latest_px = prices.iloc[-1]
        for i, t in enumerate(picks):
            target_val_usd = (total_inv / current_fx if base_currency == "KRW (₩)" else total_inv) * ai_weights[i]
            price = latest_px[t]
            qty = (target_val_usd * current_fx / price) if ".KS" in t else (target_val_usd / price)
            
            reb_list.append({
                "종목명": TICKER_INFO[t]['name'],
                "섹터": TICKER_INFO[t]['sector'],
                "티커": t,
                "비중": f"{ai_weights[i]*100:.1f}%",
                "목표수량": f"{int(qty)}주"
            })
        st.dataframe(pd.DataFrame(reb_list), use_container_width=True, hide_index=True)
        
        if st.button("🔄 포트폴리오 종목 재선정", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

# --- Footer ---
st.divider()
st.caption("Pro Tip: 상관관계가 낮은 자산(예: 주식과 금)을 혼합하면 MDD를 낮출 수 있습니다. 위 상관계수 표에서 파란색은 양의 상관관계, 붉은색은 음의 상관관계를 의미합니다.")
