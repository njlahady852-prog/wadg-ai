# ============================================================
# 🌧️ الودق AI — محرك المراقبة والتنبيهات
# مستقل عن Google Colab
# تنبيهات فقط — لا ينفذ أي صفقات
# ============================================================

import os
import time
import logging
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, timezone

# ---------------- إعداد السجلات ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------- الإعدادات ----------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

CAPITAL_SAR = float(os.getenv("CAPITAL_SAR", "10000"))
INTERVAL_MINUTES = max(
    15, int(os.getenv("CHECK_INTERVAL_MINUTES", "15"))
)

# الأسهم الأمريكية وبعض الأصول العالمية
US_ASSETS = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN",
    "GOOGL", "META", "QQQ", "SPY",
    "ARAY", "SLXN", "SOAR", "GPRO", "ARBE",
    "WKHS", "HIMX", "GRYP", "OPEN", "BBAI",
    "XYZ", "AEO", "WRD", "INMB", "LCID",
    "TOST", "INO"
]

# مؤشرات وسلع عالمية
GLOBAL_ASSETS = [
    "GC=F", "SI=F", "CL=F", "BZ=F",
    "BTC-USD", "^GSPC", "^IXIC", "^DJI"
]

# أضيفي رموز السوق السعودي هنا عند الحاجة.
# أمثلة: 2222.SR و 1120.SR و 7010.SR
SAUDI_ASSETS = [
    "2222.SR", "1120.SR", "7010.SR"
]

TICKERS = list(dict.fromkeys(
    US_ASSETS + GLOBAL_ASSETS + SAUDI_ASSETS
))

# حد مبدئي لمقدار رأس المال المخصص للمركز الواحد.
# هذا حد تخطيطي، وليس توصية استثمارية.
MAX_POSITION_PCT = 0.20

# منع تكرار الرسالة نفسها خلال دورة الفحص التالية.
last_alerts = {}


# ---------------- تيليجرام ----------------

def telegram_send(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning(
            "بيانات تيليجرام غير مكتملة. "
            "أضيفي المتغيرات في إعدادات Render."
        )
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=20
        )

        if response.ok:
            logging.info("تم إرسال رسالة تيليجرام.")
            return True

        logging.error(
            "فشل تيليجرام: %s", response.text[:500]
        )
        return False

    except requests.RequestException:
        logging.exception("تعذر الاتصال بتيليجرام.")
        return False


# ---------------- التحليل الفني ----------------

def analyze_asset(ticker):
    try:
        data = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return {
                "ticker": ticker,
                "status": "no_data"
            }

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            return {
                "ticker": ticker,
                "status": "missing_close"
            }

        close = pd.to_numeric(
            data["Close"], errors="coerce"
        ).dropna()

        if len(close) < 25:
            return {
                "ticker": ticker,
                "status": "insufficient_data"
            }

        price = float(close.iloc[-1])
        previous = float(close.iloc[-2])

        if previous == 0:
            return {
                "ticker": ticker,
                "status": "invalid_previous_price"
            }

        change_pct = (
            (price - previous) / previous
        ) * 100

        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = (
            float(close.rolling(50).mean().iloc[-1])
            if len(close) >= 50 else None
        )

        delta = close.diff()

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.rolling(14).mean().iloc[-1]
        avg_loss = losses.rolling(14).mean().iloc[-1]

        if pd.isna(avg_gain) or pd.isna(avg_loss):
            rsi = None
        elif avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = float(100 - (100 / (1 + rs)))

        if price > sma20 and (
            sma50 is None or sma20 >= sma50
        ):
            trend = "صاعد"
        elif price < sma20:
            trend = "هابط"
        else:
            trend = "متذبذب"

        # شروط تعليمية أولية للتنبيه فقط.
        # ليست ضمانًا للربح أو توصية شراء مؤكدة.
        if (
            trend == "صاعد"
            and rsi is not None
            and 40 <= rsi <= 65
        ):
            signal = "فرصة تستحق المراقبة"
        elif (
            rsi is not None
            and rsi >= 75
        ):
            signal = "ارتفاع قد يكون مبالغًا فيه"
        elif trend == "هابط":
            signal = "انتظار ومراقبة المخاطر"
        else:
            signal = "لا توجد إشارة واضحة"

        return {
            "ticker": ticker,
            "status": "ok",
            "price": price,
            "change_pct": change_pct,
            "rsi": rsi,
            "sma20": sma20,
            "sma50": sma50,
            "trend": trend,
            "signal": signal,
            "date": datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")
        }

    except Exception:
        logging.exception(
            "فشل تحليل الأصل %s", ticker
        )
        return {
            "ticker": ticker,
            "status": "error"
        }


# ---------------- دورة المراقبة ----------------

def run_scan():
    logging.info(
        "بدء دورة الفحص: %s أصل",
        len(TICKERS)
    )

    successful = 0
    alerts_sent = 0

    for ticker in TICKERS:
        result = analyze_asset(ticker)

        if result["status"] != "ok":
            logging.warning(
                "%s: لم تكتمل البيانات (%s)",
                ticker,
                result["status"]
            )
            continue

        successful += 1

        signal = result["signal"]
        price = result["price"]
        change = result["change_pct"]
        rsi = result["rsi"]

        logging.info(
            "%s | السعر %.4f | التغير %.2f%% | "
            "RSI %s | %s",
            ticker,
            price,
            change,
            f"{rsi:.1f}" if rsi is not None else "غير متاح",
            signal
        )

        # إرسال التنبيه عند وجود إشارة تستحق المتابعة.
        if signal == "فرصة تستحق المراقبة":
            alert_key = (
                ticker,
                signal,
                round(price, 4)
            )

            if last_alerts.get(ticker) != alert_key:
                message = (
                    f"🌧️ الودق AI\n"
                    f"🔔 تنبيه مراقبة فنية\n\n"
                    f"الأصل: {ticker}\n"
                    f"السعر المرصود: {price:.4f}\n"
                    f"التغير اليومي: {change:.2f}%\n"
                    f"RSI: {rsi:.1f}\n"
                    f"الاتجاه: {result['trend']}\n"
                    f"الإشارة: {signal}\n"
                    f"وقت الفحص: {result['date']}\n\n"
                    f"⚠️ تنبيه آلي أولي، وليس توصية "
                    f"شراء مؤكدة. تحققي من السعر "
                    f"والسيولة والمخاطر قبل أي قرار.\n"
                    f"لا ينفذ البوت أي صفقات."
                )

                if telegram_send(message):
                    last_alerts[ticker] = alert_key
                    alerts_sent += 1

        # التخفيف من كثرة طلبات البيانات.
        time.sleep(1)

    logging.info(
        "انتهى الفحص. بيانات ناجحة: %s، "
        "تنبيهات مرسلة: %s",
        successful,
        alerts_sent
    )


# ---------------- التشغيل الرئيسي ----------------

def main():
    logging.info("🌧️ بدء تشغيل الودق AI")

    telegram_send(
        "🌧️ الودق AI\n"
        "بدأ تشغيل محرك المراقبة.\n"
        f"عدد الأصول: {len(TICKERS)}\n"
        f"رأس المال المسجل: {CAPITAL_SAR:,.2f} ريال\n"
        f"الفحص كل {INTERVAL_MINUTES} دقيقة.\n"
        "الوضع: تنبيهات فقط، دون تداول آلي."
    )

    while True:
        try:
            run_scan()
        except Exception:
            logging.exception(
                "حدث خطأ غير متوقع في دورة الفحص."
            )

        time.sleep(INTERVAL_MINUTES * 60)

if _name_ == "_main_":
    main()
