# -*- coding: utf-8 -*-
"""
株価高騰予兆スキャナー (Streamlit版・スマホ対応)
起動: streamlit run surge_scanner_app.py
必要ライブラリ: streamlit, yfinance, pandas
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import yfinance as yf

DEFAULT_TICKERS = [
    "7203.T", "6758.T", "9984.T", "8035.T", "6861.T",
    "AAPL", "NVDA", "MSFT", "AMZN", "TSLA",
]

# ---------------------------------------------------------------- スコア計算


def calc_scores(df: pd.DataFrame) -> dict:
    """日足OHLCV DataFrame(昇順)からスコアを計算する"""
    res = {
        "volume": 0, "cross": 0, "squeeze": 0, "high52": 0, "momentum": 0,
        "total": 0, "signals": [],
    }
    if df is None or len(df) < 60:
        res["signals"].append("データ不足")
        return res

    close = df["Close"]
    volume = df["Volume"]

    # --- 1. 出来高急増 (25点) ---
    v_today = float(volume.iloc[-1])
    v_avg5 = float(volume.iloc[-6:-1].mean())
    if v_avg5 > 0:
        ratio = v_today / v_avg5
        if ratio >= 3.0:
            res["volume"] = 25
            res["signals"].append(f"出来高{ratio:.1f}倍に急増")
        elif ratio >= 2.0:
            res["volume"] = 15
            res["signals"].append(f"出来高{ratio:.1f}倍")
        elif ratio >= 1.5:
            res["volume"] = 8

    # --- 2. ゴールデンクロス (20点) ---
    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    diff = ma5 - ma25
    crossed = False
    for i in range(1, 4):  # 直近3日以内にクロス
        if len(diff) > i and diff.iloc[-i] > 0 and diff.iloc[-i - 1] <= 0:
            crossed = True
            break
    if crossed:
        res["cross"] = 20
        res["signals"].append("ゴールデンクロス発生")
    elif diff.iloc[-1] > 0:
        res["cross"] = 10

    # --- 3. ボリンジャー スクイーズ→拡張 (20点) ---
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bandwidth = (std20 * 4) / ma20
    bw = bandwidth.dropna()
    if len(bw) >= 60:
        lookback = bw.iloc[-120:] if len(bw) >= 120 else bw
        threshold = lookback.quantile(0.2)
        recent_squeeze = bool((bw.iloc[-10:] <= threshold).any())
        expanding = bw.iloc[-1] > bw.iloc[-3]
        above_ma = close.iloc[-1] > ma20.iloc[-1]
        if recent_squeeze and expanding and above_ma:
            res["squeeze"] = 20
            res["signals"].append("スクイーズ→上放れ拡張")
        elif recent_squeeze:
            res["squeeze"] = 8
            res["signals"].append("バンド収縮中(蓄積)")

    # --- 4. 52週高値 (20点) ---
    high52 = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
    last = float(close.iloc[-1])
    if last >= high52:
        res["high52"] = 20
        res["signals"].append("52週高値ブレイク")
    elif last >= high52 * 0.97:
        res["high52"] = 12
        res["signals"].append("52週高値まで3%以内")
    elif last >= high52 * 0.90:
        res["high52"] = 5

    # --- 5. モメンタム (15点) ---
    if len(close) >= 21:
        ret20 = last / float(close.iloc[-21]) - 1
        if ret20 >= 0.10:
            res["momentum"] = 15
            res["signals"].append(f"20日騰落率 +{ret20*100:.1f}%")
        elif ret20 >= 0.05:
            res["momentum"] = 10
        elif ret20 > 0:
            res["momentum"] = 5

    res["total"] = (res["volume"] + res["cross"] + res["squeeze"]
                    + res["high52"] + res["momentum"])
    return res


@st.cache_data(ttl=900, show_spinner=False)  # 15分キャッシュ
def fetch_and_score(ticker: str) -> dict:
    """1銘柄のデータ取得+スコア計算"""
    row = {
        "ticker": ticker, "name": "", "price": None, "change": None,
        "volume": 0, "cross": 0, "squeeze": 0, "high52": 0, "momentum": 0,
        "total": 0, "signals": [], "error": None,
    }
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            row["error"] = "データ取得失敗"
            return row
        try:
            row["name"] = t.info.get("shortName") or t.info.get("longName") or ""
        except Exception:
            row["name"] = ""
        row["price"] = float(df["Close"].iloc[-1])
        if len(df) >= 2:
            row["change"] = float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
        row.update(calc_scores(df))
    except Exception as e:
        row["error"] = str(e)[:80]
    return row


# ---------------------------------------------------------------- UI

st.set_page_config(page_title="高騰予兆スキャナー", page_icon="📈",
                   layout="centered")

st.title("📈 株価高騰予兆スキャナー")
st.caption("出来高・GC・BBスクイーズ・52週高値・モメンタムの5要素を100点満点でスコア化")

with st.expander("ウォッチリスト編集(1行1銘柄・日本株は 7203.T 形式)", expanded=False):
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = "\n".join(DEFAULT_TICKERS)
    st.session_state.watchlist = st.text_area(
        "銘柄リスト", st.session_state.watchlist, height=220,
        label_visibility="collapsed")

tickers = [t.strip().upper() for t in st.session_state.watchlist.splitlines()
           if t.strip()]

if st.button(f"▶ スキャン実行({len(tickers)}銘柄)", type="primary",
             use_container_width=True):
    results = []
    progress = st.progress(0.0, text="取得中...")
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(fetch_and_score, t) for t in tickers]
        for i, f in enumerate(as_completed(futures), start=1):
            results.append(f.result())
            progress.progress(i / len(tickers), text=f"{i} / {len(tickers)}")
    progress.empty()
    st.session_state.results = results

if "results" in st.session_state:
    results = sorted(st.session_state.results,
                     key=lambda x: x["total"], reverse=True)
    ok = [r for r in results if not r["error"]]
    errs = [r for r in results if r["error"]]

    # --- 上位はカード表示(スマホで見やすい) ---
    st.subheader("注目銘柄")
    hot = [r for r in ok if r["total"] >= 50]
    if hot:
        for r in hot:
            icon = "🔥" if r["total"] >= 70 else "⭐"
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{icon} {r['ticker']}** {r['name']}")
                c2.markdown(f"### {r['total']}点")
                chg = f"{r['change']:+.2f}%" if r["change"] is not None else "-"
                c1.caption(f"株価 {r['price']:,.1f}(前日比 {chg})")
                if r["signals"]:
                    st.write(" / ".join(f"`{s}`" for s in r["signals"]))
    else:
        st.info("50点以上の銘柄はありませんでした")

    # --- 全結果テーブル ---
    st.subheader("全スキャン結果")
    table = pd.DataFrame([{
        "ティッカー": r["ticker"], "銘柄名": r["name"],
        "株価": r["price"],
        "前日比%": round(r["change"], 2) if r["change"] is not None else None,
        "出来高": r["volume"], "GC": r["cross"], "BB": r["squeeze"],
        "52週": r["high52"], "勢い": r["momentum"], "合計": r["total"],
        "シグナル": " / ".join(r["signals"]),
    } for r in ok])
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.download_button(
        "CSVダウンロード",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name="surge_scan_result.csv", mime="text/csv",
        use_container_width=True)

    if errs:
        with st.expander(f"取得エラー({len(errs)}件)"):
            for r in errs:
                st.write(f"{r['ticker']}: {r['error']}")

st.caption("※本ツールはシグナル検知の補助であり、投資判断はご自身の責任でお願いします。")
