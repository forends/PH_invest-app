import streamlit as st

# ==============================
# 기본 포트폴리오 세트
# ==============================
DEFAULT_PORT = ["QQQ", "SPY", "SCHD", "TLT"]
AGGRESSIVE_PORT = ["TQQQ", "SOXL", "UPRO"]
DIVIDEND_PORT = ["SCHD", "VYM", "HDV"]

# ==============================
# 세션 상태 초기화
# ==============================
if "recommended" not in st.session_state:
    st.session_state.recommended = DEFAULT_PORT.copy()

if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# 제목
# ==============================
st.title("📊 ETF 포트폴리오 추천기")

# ==============================
# 현재 보유 종목 표시
# ==============================
st.subheader("현재 추천 종목")
st.write(st.session_state.recommended)


# ==============================
# 상태 저장 함수 (Undo용)
# ==============================
def save_history():
    st.session_state.history.append(st.session_state.recommended.copy())


# ==============================
# 리셋 / 변경 버튼 구역
# ==============================
st.subheader("포트폴리오 관리")

col1, col2 = st.columns(2)

# 🔄 기본형 복구
if col1.button("🔄 기본형"):
    save_history()
    st.session_state.recommended = DEFAULT_PORT.copy()
    st.success("기본 포트폴리오로 변경되었습니다.")
    st.rerun()

# 🚀 공격형
if col2.button("🚀 공격형"):
    save_history()
    st.session_state.recommended = AGGRESSIVE_PORT.copy()
    st.success("공격형 포트폴리오로 변경되었습니다.")
    st.rerun()


col3, col4 = st.columns(2)

# 💰 배당형
if col3.button("💰 배당형"):
    save_history()
    st.session_state.recommended = DIVIDEND_PORT.copy()
    st.success("배당형 포트폴리오로 변경되었습니다.")
    st.rerun()

# ❌ 전체 삭제
if col4.button("❌ 전체 삭제"):
    save_history()
    st.session_state.recommended = []
    st.warning("모든 종목이 제거되었습니다.")
    st.rerun()


# ==============================
# ↩ 이전 상태 복구
# ==============================
if st.button("↩ 이전 상태로 되돌리기"):
    if st.session_state.history:
        st.session_state.recommended = st.session_state.history.pop()
        st.info("이전 포트폴리오로 복구되었습니다.")
        st.rerun()
    else:
        st.error("되돌릴 기록이 없습니다.")


# ==============================
# 종목 직접 추가
# ==============================
st.subheader("종목 추가")

new_item = st.text_input("추가할 ETF 티커 입력")

if st.button("➕ 종목 추가"):
    if new_item:
        save_history()
        st.session_state.recommended.append(new_item.upper())
        st.success(f"{new_item.upper()} 추가 완료!")
        st.rerun()
    else:
        st.error("티커를 입력하세요.")


# ==============================
# 알림 영역
# ==============================
st.sidebar.header("📢 알림")
st.sidebar.write("포트폴리오 변경 시 메시지가 표시됩니다.")
