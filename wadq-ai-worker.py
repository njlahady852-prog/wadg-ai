# ================================================================
# 🌧️ الودق AI — محرك المراقبة المستمر (تنبيهات فقط، بلا تداول آلي)
# يشغّل على خدمة سحابية دائمة مثل Background Worker.
# متطلبات: pip install yfinance pandas numpy requests
#
# متغيرات البيئة المطلوبة:
# TELEGRAM_BOT_TOKEN = رمز بوت تيليجرام (سرّي)
# TELEGRAM_CHAT_ID   = رقم المحادثة
# CAPITAL_SAR       = رأس المال بالريال (اختياري، الافتراضي 10000)
# CHECK_INTERVAL_MINUTES = فترة الفحص بالدقائق (اختياري، الافتراضي 15)
# ================================================================

import os
import time
import json
import sqlite3
import logging
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- إعدادات التشغيل ----------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
CAPITAL_SAR_RAW = os.getenv("CAPITAL_SAR", "").strip()
CAPITAL_SAR = float(CAPITAL_SAR_RAW) if CAPITAL_SAR_RAW else None
RISK_PCT = min(0.05, max(0.0, float(os.getenv("RISK_PCT", "0.05"))))  # الحد الأعلى 5% من رأس المال
MAX_POSITION_PCT = min(1.0, max(0.0, float(os.getenv("MAX_POSITION_PCT", "1.00"))))  # الحد الأعلى 100% من رأس المال
USD_TO_SAR = 3.75
INTERVAL_MINUTES = max(5, int(os.getenv("CHECK_INTERVAL_MINUTES", "15")))
DB_PATH = os.getenv("WADQ_DB_PATH", "wadq_alerts.sqlite3")

# أصول المشروع: الأسهم السعودية والأمريكية المضافة
TICKERS = [
    # السعودية
    "2222.SR", "1180.SR", "1211.SR", "2010.SR", "2030.SR",
    "2380.SR", "7010.SR", "7020.SR", "7203.SR", "1120.SR",
    "1050.SR", "1150.SR", "2082.SR", "2083.SR", "2280.SR",
    # أمريكية/صناديق من المشروع
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "QQQ", "SPY", "ARAY", "SLXN", "SOAR", "GPRO", "ARBE",
    "WKHS", "HIMX", "GRYP", "OPEN", "BBAI", "XYZ", "AEO",
    "WRD", "INMB", "LCID", "TOST", "INO",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------- قاعدة لمنع تكرار التنبيه ----------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                ticker TEXT PRIMARY KEY,
                last_signal TEXT,
                last_alert_at TEXT
            )
        """)
        con.commit()

def previous_signal(ticker):
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT last_signal FROM alerts WHERE ticker=?",
            (ticker,)
        ).fetchone()
    return row[0] if row else None

def save_signal(ticker, signal):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO alerts(ticker, last_signal, last_alert_at)
            VALUES(?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                last_signal=excluded.last_signal,
                last_alert_at=excluded.last_alert_at
        """, (ticker, signal, datetime.now(timezone.utc).isoformat()))
        con.commit()

# ---------------- تيليجرام ----------------
def telegram_send(message):
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("لم يتم ضبط TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID؛ لن تُرسل الرسائل.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message},
            timeout=20
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            logging.error("Telegram API: %s", payload)
            return False
        return True
    except Exception as exc:
        logging.exception("فشل إرسال تيليجرام: %s", exc)
        return False

# ---------------- المؤشرات والتحليل ----------------
def rsi(series, period=14):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def fetch_one(ticker):
    try:
        df = yf.download(
            ticker, period="1y", interval="1d",
            auto_adjust=True, progress=False, threads=False
        )
        if df is None or df.empty:
            return None

        # yfinance may return MultiIndex even for one ticker
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                return None

        df.columns = [str(c).title() for c in df.columns]
        required = {"Close", "High", "Low"}
        if not required.issubset(df.columns):
            return None

        df = df.dropna(subset=["Close", "High", "Low"])
        if len(df) < 200:
            return None

        close = df["Close"]
        df["SMA20"] = close.rolling(20).mean()
        df["SMA50"] = close.rolling(50).mean()
        df["SMA200"] = close.rolling(200).mean()
        df["RSI14"] = rsi(close)
        df["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

        prev_close = close.shift(1)
        true_range = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        df["ATR14"] = true_range.rolling(14).mean()
        df["DAILY_CHANGE_PCT"] = close.pct_change() * 100

        if "Volume" in df.columns:
            avg_vol = df["Volume"].rolling(20).mean()
            df["VOLUME_RATIO"] = df["Volume"] / avg_vol.replace(0, np.nan)
        else:
            df["VOLUME_RATIO"] = np.nan

        df = df.dropna(subset=["SMA20", "SMA50", "SMA200", "RSI14", "MACD", "MACD_SIGNAL", "ATR14"])
        if len(df) < 2:
            return None

        last = df.iloc[-1]
        date = pd.Timestamp(df.index[-1])
        if date.tzinfo is not None:
            date = date.tz_localize(None)
        age_days = (pd.Timestamp.today().normalize() - date.normalize()).days
        fresh = age_days <= 5

        price = float(last["Close"])
        sma20, sma50, sma200 = float(last["SMA20"]), float(last["SMA50"]), float(last["SMA200"])
        macd, macd_sig = float(last["MACD"]), float(last["MACD_SIGNAL"])
        rsi_value, atr = float(last["RSI14"]), float(last["ATR14"])
        daily_change = float(last["DAILY_CHANGE_PCT"])
        vol_ratio = last.get("VOLUME_RATIO", np.nan)

        score = 0
        score += 15 if price > sma20 else 0
        score += 15 if sma20 > sma50 else 0
        score += 15 if sma50 > sma200 else 0
        score += 15 if macd > macd_sig else 0
        score += 15 if 50 <= rsi_value <= 68 else (5 if 40 <= rsi_value < 50 else 0)
        score += 10 if pd.notna(vol_ratio) and vol_ratio >= 1.2 else 0
        score += 5 if daily_change > 0 else 0
        score = min(score, 100)

        if price > sma20 and sma20 > sma50:
            trend = "صاعد"
        elif price < sma20 and sma20 < sma50:
            trend = "هابط"
        else:
            trend = "متذبذب"

        # وقف الخسارة/الأهداف: ATR، لا ضمان لنتيجة الصفقة
        stop = price - 1.5 * atr
        target1 = price + 2 * atr
        target2 = price + 3 * atr

        is_saudi = ticker.endswith(".SR")
        currency_rate = 1.0 if is_saudi else USD_TO_SAR
        market = "السوق السعودي" if is_saudi else "السوق الأمريكي"
        risk_budget_sar = CAPITAL_SAR * RISK_PCT
        max_position_sar = CAPITAL_SAR * MAX_POSITION_PCT
        stop_distance_sar = (price - stop) * currency_rate
        price_sar = price * currency_rate

        qty_risk = int(risk_budget_sar / stop_distance_sar) if stop_distance_sar > 0 else 0
        qty_position = int(max_position_sar / price_sar) if price_sar > 0 else 0
        quantity = max(0, min(qty_risk, qty_position))
        risk_sar = quantity * stop_distance_sar
        position_sar = quantity * price_sar

        if not fresh:
            signal = "بيانات قديمة - لا توصية"
        elif (
            score >= 75 and price > sma20 and 50 <= rsi_value <= 68
            and macd > macd_sig and quantity > 0
        ):
            signal = "مرشح شراء - انتظار التأكيد"
        elif score >= 50:
            signal = "مراقبة - لا شراء الآن"
        else:
            signal = "تجنب الدخول"

        return {
            "ticker": ticker, "market": market, "price": price,
            "date": date.strftime("%Y-%m-%d"), "age": age_days,
            "change": daily_change, "rsi": rsi_value, "trend": trend,
            "score": score, "signal": signal, "stop": stop,
            "target1": target1, "target2": target2,
            "qty": quantity, "position_sar": position_sar,
            "risk_sar": risk_sar, "currency_rate": currency_rate
        }
    except Exception as exc:
        logging.warning("%s: تعذر التحليل (%s)", ticker, str(exc)[:180])
        return None

def format_alert(row):
    return (
        "🌧️ الودق AI — تنبيه توصية\n"
        f"السهم: {row['ticker']} ({row['market']})\n"
        f"الإشارة: {row['signal']}\n"
        f"السعر الأخير: {row['price']:.4f}\n"
        f"تاريخ البيانات: {row['date']}\n"
        f"التغير اليومي: {row['change']:+.2f}%\n"
        f"الاتجاه: {row['trend']} | الدرجة الفنية: {row['score']}/100\n"
        f"وقف خسارة نظري: {row['stop']:.4f}\n"
        f"الأهداف النظرية: {row['target1']:.4f} / {row['target2']:.4f}\n"
        f"كمية نظرية: {row['qty']} | قيمة تقريبية: {row['position_sar']:.2f} ريال\n"
        f"مخاطرة نظرية: {row['risk_sar']:.2f} ريال\n\n"
        "⚠️ معلومات تحليلية وليست ضمانًا أو توصية مالية شخصية. "
        "لا ينفذ البوت أي شراء أو بيع."
    )

def run_scan():
    logging.info("بدء فحص %d أصلًا", len(TICKERS))
    results = []
    for ticker in TICKERS:
        row = fetch_one(ticker)
        if row:
            results.append(row)
            logging.info("%s | %s | score=%s", ticker, row["signal"], row["score"])

            old = previous_signal(ticker)
            # تنبيه عند ظهور مرشح شراء لأول مرة أو تغيّر الإشارة إلى مرشح شراء
            if row["signal"] == "مرشح شراء - انتظار التأكيد" and old != row["signal"]:
                if telegram_send(format_alert(row)):
                    save_signal(ticker, row["signal"])
            else:
                # حفظ الإشارة الحالية حتى لا يرسل تنبيهًا عند تكرار نفس الحالة
                save_signal(ticker, row["signal"])

    # إرسال ملخص فقط عند وجود مرشحين جدد؟ التنبيهات التفصيلية ترسل أعلاه.
    logging.info("انتهى الفحص: تم تحليل %d أصلًا", len(results))

def main():
    if not BOT_TOKEN or not CHAT_ID:
        raise SystemExit(
            "ضعي TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID في إعدادات الخدمة السحابية قبل التشغيل."
        )
    init_db()
    telegram_send(
        "🌧️ الودق AI بدأ العمل.\n"
        f"عدد الأصول: {len(TICKERS)}\n"
        f"رأس المال المُدخل: {CAPITAL_SAR:,.2f} ريال\n"
        f"المراقبة كل {INTERVAL_MINUTES} دقيقة.\n"
        "تنبيهات وتحليل فقط؛ لا يوجد تداول آلي."
    )
    while True:
        try:
            run_scan()
        except Exception:
            logging.exception("حدث خطأ في دورة الفحص")
        time.sleep(INTERVAL_MINUTES * 60)

if _name_ == "_main_":
main()
