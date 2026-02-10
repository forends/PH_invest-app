import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("📊 AI Portfolio Advisor Pro")

# -----------------------------
# 1. 추천 종목 데이터
# -----------------------------
@st.cache_data
def load_recommendations():
    data = {
        "Ticker": [
            "QQQ","SPY","VTI","VXUS","BND",
            "SOXX","SCHD","VIG","ARKK","VNQ"
        ],
        "ExpectedReturn": [12, 9, 8, 7, 3, 15, 8, 7, 18, 6],
        "Risk": [
            "Medium","Low","Low","Medium","Very Low",
            "High","Low","Low","Very High","Medium"
        ],
        "Reason": [
            "나스닥 기술주 성장",
            "미국 대표지수 추종",
            "미국 전체시장 분산",
            "글로벌 분산",
            "채권 안정성",
            "반도체 집중 투자",
            "배당 + 가치주",
            "배당 성장주",
            "혁신 기술 투자",
            "리츠 배당"
        ]
    }
    return pd.DataFrame(data)

df = load_recommendations()

# -----------------------------
# 2. 비중 자동 계산 (리스크 기반)
# -----------------------------
risk_score = {
    "Very Low": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
    "Very High": 5,
}

df["RiskScore"] = df["Risk"].map(risk_score)

# 위험 낮을수록 높은 비중
df["Weight"] = (1 / df["RiskScore"])
df["Weight"] = df["Weight"] / df["Weight"].sum() * 100

# -----------------------------
# 3. 기대 수익률 계산
# -----------------------------
port_return = np.sum(df["ExpectedReturn"] * df["Weight"] / 100)

# NaN, None 방지
if pd.isna(port_return):
    port_return = 0.0

# -----------------------------
# 4. 화면 좌우 분할
# -----------------------------
left, right = st.columns(2)

# -----------------------------
# LEFT : 안정 포트폴리오 + 추천 비중
# -----------------------------
with left:
    st.subheader("📦 추천 포트폴리오 & 비중")

    for _, row in df.iterrows():

        risk_color = {
            "Very Low": "🟢",
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🟠",
            "Very High": "🔴",
        }[row["Risk"]]

        st.markdown(
            f"""
            **{row['Ticker']}**  
            비중: **{row['Weight']:.1f}%**  
            기대수익률: **{row['ExpectedReturn']}%**  
            위험도: {risk_color} {row['Risk']}  
            이유: {row['Reason']}
            """
        )
        st.divider()

# -----------------------------
# RIGHT : 포트폴리오 요약
# -----------------------------
with right:
    st.subheader("📈 포트폴리오 요약")

    k1, k2 = st.columns(2)

    k1.metric("Expected Return (1Y)", f"{float(port_return):.2f}%")
    k2.metric("종목 수", len(df))

    # 목표 수익 설정
    st.subheader("🎯 목표 수익 알림")

    target = st.slider("목표 수익률 (%)", 5, 30, 15)

    if port_return >= target:
        st.success("목표 수익률 달성 가능성이 있습니다!")
    else:
        st.info("현재 기준으로 목표 수익에 조금 부족합니다.")

    # 리밸런싱 체크
    st.subheader("🔄 리밸런싱 추천")

    high_risk_ratio = df[df["RiskScore"] >= 4]["Weight"].sum()

    if high_risk_ratio > 40:
        st.warning("고위험 자산 비중이 높습니다. 일부를 채권/배당 ETF로 이동 추천.")
    else:
        st.success("리스크 균형이 적절합니다.")

# -----------------------------
# 5. 리셋 버튼
# -----------------------------
if st.button("🔄 추천 포트폴리오 리셋"):
    st.cache_data.clear()
    st.experimental_rerun()
