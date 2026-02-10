import streamlit as st
import yfinance as yf
import pandas as pd

st.title("🚀 공격형 포트폴리오 운용 시스템 PRO")

# =====================================================
# 투자금 & 보유 수량 입력
# =====================================================
st.header("💼 내 보유 현황 입력")

tickers = ["QQQ", "SOXL", "TQQQ", "SMH", "NVDA", "TSLA"]

qty_dict = {}

for t in tickers:
    qty_dict[t] = st.number_input(f"{t} 보유 수량", value=0)

# =====================================================
# 목표 비중
# =====================================================
weights = {
    "QQQ": 0.25,
    "SOXL": 0.20,
    "TQQQ": 0.15,
    "SMH": 0.15,
    "NVDA": 0.15,
    "TSLA": 0.10
}

# =====================================================
# 실행 버튼
# =====================================================
if st.button("분석 시작"):

    raw = yf.download(tickers, period="1y", auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        data = raw

    price = data.iloc[-1]

    # =================================================
    # 현재 평가금액 계산
    # =================================================
    current_values = {t: qty_dict[t] * price[t] for t in tickers}
    total_money = sum(current_values.values())

    st.subheader("💰 현재 평가금액")
    st.write(f"총 자산: ${int(total_money):,}")

    df_now = pd.DataFrame(
        [[t, qty_dict[t], round(price[t],2), int(current_values[t])] for t in tickers],
        columns=["종목", "보유수량", "현재가", "평가금액"]
    )
    st.dataframe(df_now)

    # =================================================
    # 수익률 (최근 3개월)
    # =================================================
    ret = data.pct_change(63).iloc[-1]
    st.subheader("📈 3개월 수익률")
    st.dataframe(ret.sort_values(ascending=False))

    # =================================================
    # 시장 위험 판단 (QQQ 기준)
    # =================================================
    ma50 = data.rolling(50).mean().iloc[-1]["QQQ"]
    ma200 = data.rolling(200).mean().iloc[-1]["QQQ"]

    st.subheader("🚨 시장 위험 신호")

    if ma50 < ma200:
        st.error("하락장 가능성 ↑ 레버리지 비중 줄이기 권장")
    else:
        st.success("상승 추세 👍 공격적 운용 가능")

    # =================================================
    # 리밸런싱 계산
    # =================================================
    st.subheader("🎯 리밸런싱 매매 제안")

    orders = []

    for t in tickers:
        target_amount = total_money * weights[t]
        diff_money = target_amount - current_values[t]
        qty = diff_money / price[t]

        if diff_money > 0:
            action = "매수"
        elif diff_money < 0:
            action = "매도"
        else:
            action = "유지"

        orders.append([t, action, int(abs(diff_money)), int(abs(qty))])

    df_orders = pd.DataFrame(
        orders,
        columns=["종목", "액션", "주문금액($)", "주문수량(주)"]
    )

    st.dataframe(df_orders)
