import requests
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import logging
from datetime import datetime, timedelta
import io
import os
import warnings
warnings.filterwarnings('ignore')

# ========== KONFIGURASI ==========
BOT_TOKEN = "8258953476:AAFYvnS0AJV3fHTJKTvSIxWCwrfJSWUnL4E"
SPREADSHEET_ID = "134MCh1v_puOMK2jzc9iIzcXpVdH_2hRUmPEwRZr7XPY"
GID = "442523735"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}"

# === SESUAIKAN PATH INI ===
# Jika file .pkl berada di folder yang sama dengan script, gunakan "."
# Jika di folder lain, tulis path lengkapnya, misal: r"D:\TA\OUTPUT_XGBOOST_2MODEL"
MODEL_PATH = r"D:\TA\OUTPUT_XGBOOST_2MODEL"  # <-- GANTI DENGAN PATH ANDA

CLASSIFIER_PATH = os.path.join(MODEL_PATH, "model_classifier.pkl")
REGRESSOR_PATH = os.path.join(MODEL_PATH, "model_regressor.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_PATH, "feature_cols.pkl")
THRESHOLD_PATH = os.path.join(MODEL_PATH, "threshold.pkl")
ISOTONIC_PATH = os.path.join(MODEL_PATH, "isotonic_calibrator.pkl")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== LOAD MODEL DENGAN PENGECEKAN ==========
clf = reg = feature_cols = best_thresh = iso_reg = None
model_loaded = False

def load_pkl(file_path, name):
    if not os.path.exists(file_path):
        logger.error(f"❌ File tidak ditemukan: {file_path}")
        return None
    try:
        obj = joblib.load(file_path)
        logger.info(f"✅ {name} berhasil dimuat dari {file_path}")
        return obj
    except Exception as e:
        logger.error(f"❌ Gagal load {name}: {e}")
        return None

# Cek keberadaan semua file
print("\n🔍 Memeriksa file model...")
for p in [CLASSIFIER_PATH, REGRESSOR_PATH, FEATURE_COLS_PATH, THRESHOLD_PATH]:
    print(f"  {p} -> {'✅ ADA' if os.path.exists(p) else '❌ TIDAK ADA'}")

clf = load_pkl(CLASSIFIER_PATH, "Classifier")
reg = load_pkl(REGRESSOR_PATH, "Regressor")
feature_cols = load_pkl(FEATURE_COLS_PATH, "Feature Columns")
best_thresh = load_pkl(THRESHOLD_PATH, "Threshold")
iso_reg = load_pkl(ISOTONIC_PATH, "Isotonic Calibrator")

if clf is not None and reg is not None and feature_cols is not None and best_thresh is not None:
    model_loaded = True
    logger.info("✅ Semua model berhasil dimuat. Bot siap melakukan prediksi.")
else:
    logger.warning("⚠️ Model tidak lengkap. Fitur prediksi tidak akan berfungsi.")

# ========== FUNGSI FEATURE ENGINEERING ==========
def create_features(df_in):
    d = df_in.copy()
    d['month'] = d['date'].dt.month
    d['dayofyear'] = d['date'].dt.dayofyear
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['doy_sin']   = np.sin(2 * np.pi * d['dayofyear'] / 365)
    d['doy_cos']   = np.cos(2 * np.pi * d['dayofyear'] / 365)
    d['wet_season'] = d['month'].isin([11, 12, 1, 2, 3]).astype(int)
    d['transition_season'] = d['month'].isin([4, 5, 10]).astype(int)

    for col in ['rr', 'tavg', 'rh', 'wind']:
        for lag in [1, 2, 3, 7, 14]:
            d[f'{col}_lag{lag}'] = d[col].shift(lag)
    for lag in [1, 2, 3, 7, 14]:
        d[f'log_lag_rr_{lag}'] = np.log1p(d[f'rr_lag{lag}'])

    for w in [3, 7, 14]:
        d[f'rr_roll_mean{w}'] = d['rr'].shift(1).rolling(w, min_periods=1).mean()
        d[f'rr_roll_std{w}']  = d['rr'].shift(1).rolling(w, min_periods=1).std()
        d[f'rr_roll_max{w}']  = d['rr'].shift(1).rolling(w, min_periods=1).max()
        d[f'rr_roll_sum{w}']  = d['rr'].shift(1).rolling(w, min_periods=1).sum()
        d[f'rh_roll_mean{w}'] = d['rh'].shift(1).rolling(w, min_periods=1).mean()
        d[f'tavg_roll_mean{w}'] = d['tavg'].shift(1).rolling(w, min_periods=1).mean()

    d['rr_accum3d'] = d['rr'].shift(1).rolling(3, min_periods=1).sum()
    d['rr_accum7d'] = d['rr'].shift(1).rolling(7, min_periods=1).sum()
    d['rr_ewm_7']   = d['rr'].shift(1).ewm(span=7, adjust=False).mean()

    d['rh_x_temp'] = d['rh'] * d['tavg']
    d['rh_x_wind'] = d['rh'] * d['wind']
    d['temp_x_wind'] = d['tavg'] * d['wind']

    streak = dry = 0
    rain_streak, dry_spell = [], []
    for v in d['rr'].shift(1).fillna(0):
        if v > 0.5:
            streak += 1; dry = 0
        else:
            dry += 1; streak = 0
        rain_streak.append(streak)
        dry_spell.append(dry)
    d['rain_streak'] = rain_streak
    d['dry_spell']   = dry_spell

    d['rr_trend_1'] = d['rr'] - d['rr'].shift(1)
    d['rr_trend_3'] = d['rr'] - d['rr'].shift(3)

    d['wet_lag1_rr'] = d['wet_season'] * d['rr_lag1']
    d['wet_lag2_rr'] = d['wet_season'] * d['rr_lag2']
    d['wet_lag3_rr'] = d['wet_season'] * d['rr_lag3']

    d['rh_temp_ratio'] = d['rh'] / (d['tavg'] + 0.1)
    d['heavy_rain_recent'] = (d['rr'].shift(1).rolling(3).max() > 50).astype(int)
    d['vapor_pressure'] = (d['rh'] / 100) * 6.112 * np.exp((17.67 * d['tavg']) / (d['tavg'] + 243.5))
    d['rr_streak'] = (d['rr'] > 0).astype(int)

    return d

# ========== FUNGSI PREDIKSI ==========
def prediksi_xgboost(df_hari_ini, n_days=8):
    if not model_loaded:
        return None

    df = df_hari_ini.copy()
    df = df.rename(columns={
        'suhu': 'tavg',
        'kelembaban': 'rh',
        'curah_hujan': 'rr',
        'kecepatan_angin': 'wind'
    })
    if 'date' not in df.columns:
        if 'tanggal' in df.columns:
            df.rename(columns={'tanggal': 'date'}, inplace=True)
        else:
            df['date'] = pd.date_range(end=datetime.now(), periods=len(df))

    df = df.sort_values('date').reset_index(drop=True)
    current_df = df[['date', 'rr', 'tavg', 'rh', 'wind']].copy()
    results = []

    for _ in range(n_days):
        new_date = current_df['date'].max() + timedelta(days=1)
        last_row = current_df.iloc[-1]
        new_row = pd.DataFrame({
            'date': [new_date],
            'rr': [0.0],
            'tavg': [last_row['tavg']],
            'rh': [last_row['rh']],
            'wind': [last_row['wind']]
        })
        temp_df = pd.concat([current_df, new_row], ignore_index=True)
        temp_df = create_features(temp_df)
        X_new = temp_df[feature_cols].iloc[-1:].fillna(0)

        proba = clf.predict_proba(X_new)[0, 1]
        if proba >= best_thresh:
            pred_log = reg.predict(X_new)
            if iso_reg is not None:
                pred_log = iso_reg.predict(pred_log)
            intensity = np.expm1(pred_log)[0]
            intensity = max(0.0, intensity)
        else:
            intensity = 0.0

        new_row['rr'] = intensity
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        current_df = current_df.iloc[-30:].reset_index(drop=True)
        results.append(round(intensity, 1))

    return results

# ========== FUNGSI AMBIL DATA DARI GOOGLE SHEETS ==========
def get_data_from_google_sheets():
    try:
        response = requests.get(CSV_URL, timeout=15)
        response.raise_for_status()
        content = response.text
        lines = content.splitlines()
        start_line = 0
        for i, line in enumerate(lines):
            if 'tanggal' in line.lower() or 'suhu' in line.lower():
                start_line = i
                break
        csv_content = "\n".join(lines[start_line:])
        df = pd.read_csv(io.StringIO(csv_content))
        df.columns = df.columns.str.strip().str.lower()
        for col in ['suhu', 'kelembaban', 'curah_hujan', 'kecepatan_angin']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float, errors='ignore')
        if 'tanggal' in df.columns:
            df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce', dayfirst=True)
            df = df.sort_values('tanggal')
        df = df.dropna(subset=['suhu'], how='all')
        logger.info(f"Data berhasil: {len(df)} baris")
        return df if not df.empty else None
    except Exception as e:
        logger.error(f"Error ambil data: {e}")
        return None

# ========== HANDLER TELEGRAM ==========
async def start(update, context):
    user = update.effective_user
    keyboard = [
        [InlineKeyboardButton("🌦️ Info Cuaca", callback_data='cuaca'),
         InlineKeyboardButton("🌡️ Info Suhu", callback_data='suhu')],
        [InlineKeyboardButton("📊 Prediksi 8 Hari", callback_data='prediksi'),
         InlineKeyboardButton("📈 Status Data", callback_data='status')]
    ]
    await update.message.reply_text(
        f"Halo {user.first_name}! 👋\n\nGunakan tombol di bawah untuk informasi cuaca.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def status(update, context):
    msg = await update.message.reply_text("⏳ Mengecek data...")
    df = get_data_from_google_sheets()
    if df is None or df.empty:
        await msg.edit_text("❌ Tidak ada data.\nPastikan spreadsheet dapat diakses dan berisi data.")
        return
    await msg.edit_text(f"📊 *Status Data*\nTotal baris: {len(df)}\nKolom: {', '.join(df.columns)}", parse_mode='Markdown')

async def callback_handler(update, context):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id

    if data == 'cuaca':
        await query.edit_message_text("⏳ Mengambil data cuaca...")
        df = get_data_from_google_sheets()
        if df is None or df.empty:
            await context.bot.send_message(chat_id, "❌ Tidak ada data cuaca.")
            return
        latest = df.iloc[-1]
        tgl = latest['tanggal'].strftime('%d/%m/%Y %H:%M:%S') if 'tanggal' in df.columns and pd.notna(latest.get('tanggal')) else "Data terbaru"
        pesan = f"🌤️ *Info Cuaca Terkini*\n📅 {tgl}\n"
        if 'suhu' in df.columns and pd.notna(latest.get('suhu')):
            pesan += f"🌡️ Suhu: {latest['suhu']} °C\n"
        if 'kelembaban' in df.columns and pd.notna(latest.get('kelembaban')):
            pesan += f"💧 Kelembaban: {latest['kelembaban']} %\n"
        if 'curah_hujan' in df.columns and pd.notna(latest.get('curah_hujan')):
            pesan += f"☔ Curah hujan: {latest['curah_hujan']} mm\n"
        if 'kecepatan_angin' in df.columns and pd.notna(latest.get('kecepatan_angin')):
            pesan += f"💨 Kecepatan angin: {latest['kecepatan_angin']} m/s\n"
        await context.bot.send_message(chat_id, pesan, parse_mode='Markdown')
    
    elif data == 'suhu':
        await query.edit_message_text("⏳ Mengambil data suhu...")
        df = get_data_from_google_sheets()
        if df is None or df.empty or 'suhu' not in df.columns:
            await context.bot.send_message(chat_id, "❌ Data suhu tidak tersedia.")
            return
        df_suhu = df[df['suhu'].notna()]
        if df_suhu.empty:
            await context.bot.send_message(chat_id, "❌ Belum ada data suhu.")
            return
        latest = df_suhu.iloc[-1]
        waktu = f" ({latest['tanggal'].strftime('%d/%m %H:%M')})" if 'tanggal' in df.columns and pd.notna(latest.get('tanggal')) else ""
        await context.bot.send_message(chat_id, f"🌡️ *Info Suhu Terkini*\nSuhu: {latest['suhu']} °C{waktu}", parse_mode='Markdown')
    
    elif data == 'prediksi':
        if not model_loaded:
            await context.bot.send_message(chat_id, "❌ Model prediksi belum tersedia. Pastikan file model (.pkl) sudah ditempatkan di direktori yang benar.")
            return
        await query.edit_message_text("⏳ Menghitung prediksi 8 hari dengan model XGBoost...")
        df = get_data_from_google_sheets()
        if df is None or df.empty or len(df) < 30:
            await context.bot.send_message(chat_id, "❌ Data historis tidak cukup (minimal 30 hari) untuk prediksi.")
            return
        required_cols = ['suhu', 'kelembaban', 'curah_hujan', 'kecepatan_angin']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            await context.bot.send_message(chat_id, f"❌ Kolom yang diperlukan tidak tersedia: {missing}")
            return
        df_history = df.tail(30).copy()
        hasil = prediksi_xgboost(df_history, n_days=8)
        if hasil is None:
            await context.bot.send_message(chat_id, "❌ Gagal melakukan prediksi. Periksa data atau model.")
            return
        tgl_mulai = datetime.now().date()
        pesan = "📅 *Ramalan Curah Hujan 8 Hari (XGBoost)*\n"
        for i, nilai in enumerate(hasil):
            tgl = tgl_mulai + timedelta(days=i+1)
            pesan += f"• {tgl.strftime('%d/%m')}: {nilai} mm\n"
        await context.bot.send_message(chat_id, pesan, parse_mode='Markdown')
    
    elif data == 'status':
        df = get_data_from_google_sheets()
        if df is None:
            await context.bot.send_message(chat_id, "❌ Gagal ambil data.")
            return
        await context.bot.send_message(chat_id, f"📊 Total baris: {len(df)}\nKolom: {', '.join(df.columns)}")

def main():
    if not model_loaded:
        logger.warning("⚠️ Model tidak lengkap. Fitur prediksi tidak akan berfungsi.")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CallbackQueryHandler(callback_handler))
    logger.info("Bot polling dimulai...")
    app.run_polling()

if __name__ == "__main__":
    main()