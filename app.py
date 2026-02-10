import streamlit as st
import yfinance as yf
import pandas as pd

st.title("🚀 공격형 포트폴리오 리밸런싱 시스템")

total_money = st.number_input("총 투자금 입력", value=10000000, step=1000000)

tickers = {
    "QQQ": 0.25,
    "SOXL": 0.20,
    "TQQQ": 0.15,
    "SMH": 0.15,
    "NVDA": 0.15,
    "TSLA": 0.10
}

ticker_list = list(tickers.keys())

if st.button("리밸런싱 계산 시작"):

    raw = yf.download(ticker_list, period="1y", auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        data = raw

    today_price = data.iloc[-1]

    ma50 = data.rolling(50).mean().iloc[-1]
    ma200 = data.rolling(200).mean().iloc[-1]

    trend = []
    for t in ticker_list:
        if ma50[t] > ma200[t]:
            signal = "상승추세"
        else:
            signal = "하락추세"
        trend.append([t, round(today_price[t],2), signal])

    df_trend = pd.DataFrame(trend, columns=["종목", "현재가격", "추세"])

    st.subheader("📈 시장 추세")
    st.dataframe(df_trend)

    equal_weight = total_money / len(ticker_list)

    orders = []
    for t in ticker_list:
        target_amount = total_money * tickers[t]
        diff_money = target_amount - equal_weight
        qty = diff_money / today_price[t]

        if diff_money > 0:
            action = "매수"
        elif diff_money < 0:
            action = "매도"
        else:
            action = "유지"

        orders.append([
            t,
            action,
            int(abs(diff_money)),
            int(abs(qty))
        ])

    df_orders = pd.DataFrame(
        orders,
        columns=["종목", "액션", "주문금액", "주문수량(주)"]
    )

    st.subheader("💰 매매 지시서")
    st.dataframe(df_orders)
