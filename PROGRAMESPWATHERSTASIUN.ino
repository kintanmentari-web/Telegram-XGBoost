#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Adafruit_AHTX0.h>
#include <time.h>

// ========== KONFIGURASI WIFI ==========
const char* ssid = "ggh";              // <-- PASTIKAN SSID ANDA
const char* password = "12345678";        // <-- PASTIKAN PASSWORD

// ========== GOOGLE SHEETS (URL HASIL DEPLOY) ==========
const char* scriptURL_realtime = "https://script.google.com/macros/s/AKfycbxdHc0pGOn3zWPFP9_tb0eHNArY91V7D4nfQefF61FejXX-5YLjrL8a9wbNfEPaAcplhQ/exec";
const char* scriptURL_historis = "https://script.google.com/macros/s/AKfycbyZ4Gm-iM8799z2zKRVzxKOA8qbwyMgosNWaddxi-ROfspMJXxj116DPjQkNvhmmWVqQQ/exec";

unsigned long lastSendRealtime = 0;
const unsigned long realtimeInterval = 1000UL;   // 1 DETIK
unsigned long lastSendHistoris = 0;
const unsigned long historiInterval = 21600000UL; // 1 JAM

// ========== TELEGRAM ==========
const char* botToken = "8258953476:AAFpcwQIUv7F1lHRGof4iONyRQh0WKHfjeM";
const String adminChatID = "5588161088";

// ========== SENSOR HUJAN (TIPPING BUCKET) ==========
#define PIN_HUJAN 32
#define MM_PER_TICK 1.27

const unsigned long minRainInterval = 500;
const unsigned long rainConfirmWindow = 10000;
const unsigned int minPulseToConfirm = 1;

volatile unsigned long tickHujan = 0;
volatile unsigned long lastRainPulse = 0;
unsigned long lastRainTickTime = 0;
const unsigned long resetTimeout = 60000;

unsigned long pulseTimestamps[10] = {0};
unsigned int pulseIndex = 0;
unsigned int pulseCount = 0;

float curahHujan = 0;

void IRAM_ATTR hitungHujan() {
  unsigned long now = millis();
  if (now - lastRainPulse >= minRainInterval) {
    pulseTimestamps[pulseIndex] = now;
    pulseIndex = (pulseIndex + 1) % 10;
    if (pulseCount < 10) pulseCount++;

    tickHujan++;
    lastRainPulse = now;
    lastRainTickTime = now;
  }
}

// ========== SENSOR ANGIN (ANEMOMETER) ==========
#define PIN_ANEMO 27
volatile unsigned long pulseAngin = 0;
volatile unsigned long lastWindPulse = 0;
const unsigned long minWindInterval = 2;
float kecepatanAnginRaw = 0;
float kecepatanAnginFiltered = 0;
float kecepatanAngin = 0;
const float windAlpha = 0.3;
const float WIND_FACTOR = 0.61;

void IRAM_ATTR hitungAngin() {
  unsigned long now = millis();
  if (now - lastWindPulse >= minWindInterval) {
    pulseAngin++;
    lastWindPulse = now;
  }
}

// ========== SENSOR AHT10 ==========
Adafruit_AHTX0 aht;
bool aht_ok = false;
float suhu_raw = 0, hum_raw = 0;
float suhu_terkalibrasi = 0, hum_terkalibrasi = 0;
float suhu_ema = 0, hum_ema = 0;
const float alpha = 0.2;

const float kalibrasi_suhu = 1.1;
const float kalibrasi_hum  = -8.1;

// ========== NTP ==========
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 25200;
const int daylightOffset_sec = 0;
bool timeIsSynchronized = false;

unsigned long lastSensorRead = 0;
const unsigned long sensorInterval = 1000;
unsigned long lastLog = 0;

const int jamTarget[] = {6, 17, 22};
const int menitTarget[] = {0, 0, 0};
const int jumlahJadwal = 3;
bool sudahKirimHariIni[jumlahJadwal] = {false, false, false};
int lastDay = -1;

WebServer server(80);
WiFiClientSecure clientSecure;
bool lastWiFiStatus = false;

// ========== LCD 20x4 ==========
#define LCD_ADDR 0x27   // Jika tidak muncul, coba 0x3F
LiquidCrystal_I2C lcd(LCD_ADDR, 20, 4);

// ========== PROTOTIPE ==========
void bacaSensorAHT();
void kirimDataHistoris(float suhu, float hum, float hujan, float angin);
void kirimDataRealtime(float suhu, float hum, float hujan, float angin);
void kirimPerintahPrediksi(float suhu, float hum, float hujan, float angin);
void sendTelegramMessage(String message);
String getTimestamp();
void handleSensorData();

// ========== SETUP ==========
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== WEATHER STATION (Realtime 1 detik, Historis 1 jam) ===");

  // ---- Inisialisasi LCD ----
  lcd.init();
  lcd.backlight();
  lcd.setCursor(2, 1);
  lcd.print("WEATHER STATION");
  delay(1500);
  lcd.clear();

  // Koneksi WiFi
  WiFi.begin(ssid, password);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nIP: " + WiFi.localIP().toString());

  // Sinkronisasi waktu NTP
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  Serial.print("NTP");
  int attempts = 0;
  while (!timeIsSynchronized && attempts < 30) {
    struct tm timeinfo;
    if (getLocalTime(&timeinfo, 1000)) {
      int year = timeinfo.tm_year + 1900;
      if (year > 2023 && year < 2030) {
        timeIsSynchronized = true;
        Serial.println("\nWaktu sinkron!");
        char buf[30];
        strftime(buf, 30, "%Y-%m-%d %H:%M:%S", &timeinfo);
        Serial.println(buf);
        break;
      }
    }
    attempts++;
    delay(1000);
    Serial.print(".");
  }
  if (!timeIsSynchronized) Serial.println("\nGagal NTP! Coba ganti ntpServer.");

  clientSecure.setInsecure();

  // I2C - set clock ke 100kHz untuk stabilitas
  Wire.begin(21, 22);
  Wire.setClock(100000);
  Serial.println("I2C: SDA=21, SCL=22, Clock=100kHz");

  // Interrupt sensor hujan & angin
  pinMode(PIN_HUJAN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_HUJAN), hitungHujan, FALLING);
  pinMode(PIN_ANEMO, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ANEMO), hitungAngin, FALLING);

  // Inisialisasi AHT10
  aht_ok = aht.begin();
  if (!aht_ok) Serial.println("AHT10 error!");
  else Serial.println("AHT10 OK");

  // HTTP server
  server.on("/data", handleSensorData);
  server.begin();
  Serial.println("HTTP server ready");

  lastRainTickTime = millis();
  lastWiFiStatus = true;
}

// ========== LOOP ==========
void loop() {
  bool wifiConnected = (WiFi.status() == WL_CONNECTED);
  if (wifiConnected != lastWiFiStatus) {
    if (wifiConnected) {
      lcd.backlight();
      lcd.clear();
    } else {
      lcd.noBacklight();
    }
    lastWiFiStatus = wifiConnected;
  }

  bacaSensorAHT();
  curahHujan = tickHujan * MM_PER_TICK;

  // ---------- VALIDASI PULSA HUJAN ----------
  static unsigned long lastValidation = 0;
  if (millis() - lastValidation >= 1000) {
    lastValidation = millis();
    unsigned long now = millis();
    unsigned int validPulses = 0;
    for (int i = 0; i < pulseCount; i++) {
      if (now - pulseTimestamps[i] <= rainConfirmWindow) {
        validPulses++;
      }
    }
    if (validPulses < minPulseToConfirm && tickHujan > 0) {
      noInterrupts();
      tickHujan = 0;
      interrupts();
      lastRainTickTime = millis();
      curahHujan = 0;
      Serial.println("[FILTER] Pulsa tunggal diabaikan.");
    }
  }

  if (tickHujan > 0 && (millis() - lastRainTickTime >= resetTimeout)) {
    noInterrupts();
    tickHujan = 0;
    interrupts();
    lastRainTickTime = millis();
    curahHujan = 0;
    Serial.println("[RESET] Hujan berhenti.");
  }

  // ====== ANGIN ======
  static unsigned long lastWindCalc = 0;
  if (millis() - lastWindCalc >= 1000) {
    lastWindCalc = millis();
    noInterrupts();
    unsigned long pulses = pulseAngin;
    pulseAngin = 0;
    interrupts();
    kecepatanAnginRaw = pulses * WIND_FACTOR;
    if (pulses == 0) {
      kecepatanAnginFiltered = 0;
      kecepatanAngin = 0;
    } else {
      if (kecepatanAnginFiltered == 0) kecepatanAnginFiltered = kecepatanAnginRaw;
      else kecepatanAnginFiltered = windAlpha * kecepatanAnginRaw + (1 - windAlpha) * kecepatanAnginFiltered;
      float delta = kecepatanAnginFiltered - kecepatanAngin;
      if (delta > 0.2) kecepatanAnginFiltered = kecepatanAngin + 0.2;
      if (delta < -0.2) kecepatanAnginFiltered = kecepatanAngin - 0.2;
      kecepatanAngin = kecepatanAnginFiltered;
    }
  }

  // ====== TAMPILAN LCD (setiap 500ms agar stabil) ======
  static unsigned long lastLCD = 0;
  if (wifiConnected && (millis() - lastLCD >= 500)) {
    lastLCD = millis();
    char buf[21];

    // Baris 0: Suhu dan RH
    lcd.setCursor(0, 0);
    lcd.print("                    "); // hapus baris
    lcd.setCursor(0, 0);
    snprintf(buf, 21, "T:%.1fC RH:%.1f%%", suhu_terkalibrasi, hum_terkalibrasi);
    lcd.print(buf);

    // Baris 1: Hujan
    lcd.setCursor(0, 1);
    lcd.print("                    ");
    lcd.setCursor(0, 1);
    snprintf(buf, 21, "Hujan: %.1f mm", curahHujan);
    lcd.print(buf);

    // Baris 2: Angin
    lcd.setCursor(0, 2);
    lcd.print("                    ");
    lcd.setCursor(0, 2);
    snprintf(buf, 21, "Angin: %.1f m/s", kecepatanAngin);
    lcd.print(buf);

    // Baris 3: Status WiFi
    lcd.setCursor(0, 3);
    lcd.print("WiFi OK            ");
  }

  // ====== KIRIM DATA ======
  if (wifiConnected && timeIsSynchronized) {
    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
      int jam = timeinfo.tm_hour;
      int menit = timeinfo.tm_min;
      int today = timeinfo.tm_mday;
      if (today != lastDay) {
        for (int i=0; i<jumlahJadwal; i++) sudahKirimHariIni[i] = false;
        lastDay = today;
      }
      for (int i=0; i<jumlahJadwal; i++) {
        if (jam == jamTarget[i] && menit == menitTarget[i] && !sudahKirimHariIni[i]) {
          kirimPerintahPrediksi(suhu_terkalibrasi, hum_terkalibrasi, curahHujan, kecepatanAngin);
          sudahKirimHariIni[i] = true;
        }
      }
    }

    if (millis() - lastSendRealtime >= realtimeInterval) {
      lastSendRealtime = millis();
      kirimDataRealtime(suhu_terkalibrasi, hum_terkalibrasi, curahHujan, kecepatanAngin);
    }
    if (millis() - lastSendHistoris >= historiInterval) {
      lastSendHistoris = millis();
      kirimDataHistoris(suhu_terkalibrasi, hum_terkalibrasi, curahHujan, kecepatanAngin);
    }
  }

  server.handleClient();

  // ====== LOG SERIAL ======
  if (millis() - lastLog >= 5000) {
    lastLog = millis();
    Serial.printf("Suhu: %.1f C, RH: %.1f%%, Hujan: %.1f mm, Angin: %.1f m/s, Tick: %lu, Sync: %s\n",
                  suhu_terkalibrasi, hum_terkalibrasi, curahHujan, kecepatanAngin, tickHujan,
                  timeIsSynchronized ? "YA" : "TIDAK");
  }

  delay(1);
}

// ========== BACA SENSOR AHT10 ==========
void bacaSensorAHT() {
  if (!aht_ok) return;
  if (millis() - lastSensorRead >= sensorInterval) {
    lastSensorRead = millis();
    sensors_event_t h, t;
    if (aht.getEvent(&h, &t)) {
      if (!isnan(t.temperature)) suhu_raw = t.temperature;
      if (!isnan(h.relative_humidity)) hum_raw = h.relative_humidity;
      if (suhu_ema == 0) {
        suhu_ema = suhu_raw;
        hum_ema = hum_raw;
      } else {
        suhu_ema = alpha * suhu_raw + (1 - alpha) * suhu_ema;
        hum_ema = alpha * hum_raw + (1 - alpha) * hum_ema;
      }
      suhu_terkalibrasi = constrain(suhu_ema + kalibrasi_suhu, -40, 85);
      hum_terkalibrasi = constrain(hum_ema + kalibrasi_hum, 0, 100);
    } else {
      Serial.println("Gagal baca AHT10");
      aht.begin();
    }
  }
}

// ========== AMBIL TIMESTAMP ==========
String getTimestamp() {
  if (!timeIsSynchronized) return "1970-01-01 00:00:00";
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return "1970-01-01 00:00:00";
  char buffer[20];
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(buffer);
}

// ========== KIRIM DATA KE REALTIME SHEET ==========
void kirimDataRealtime(float suhu, float hum, float hujan, float angin) {
  if (WiFi.status() != WL_CONNECTED) return;
  if (isnan(suhu) || isnan(hum) || isnan(hujan) || isnan(angin)) {
    Serial.println("[RT] Data NaN, skip");
    return;
  }
  String ts = getTimestamp();
  if (ts == "1970-01-01 00:00:00") {
    Serial.println("[RT] Timestamp invalid, skip");
    return;
  }

  HTTPClient http;
  http.begin(clientSecure, scriptURL_realtime);
  http.addHeader("Content-Type", "application/json");
  http.setUserAgent("ESP32-WeatherStation/2.0");
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  http.setRedirectLimit(3);
  http.setTimeout(8000);

  String payload = "{\"timestamp\":\"" + ts + "\",";
  payload += "\"temperature\":" + String(suhu,1) + ",";
  payload += "\"humidity\":" + String(hum,1) + ",";
  payload += "\"rainfall\":" + String(hujan,1) + ",";
  payload += "\"wind_speed\":" + String(angin,1) + "}";

  int code = http.POST(payload);
  if (code == 200) {
    // optional
  } else {
    Serial.printf("[RT] HTTP %d\n", code);
    if (code == 400) {
      String resp = http.getString();
      Serial.println("Respon error: " + resp.substring(0, 150));
    }
  }
  http.end();
}

// ========== KIRIM DATA KE HISTORIS SHEET ==========
void kirimDataHistoris(float suhu, float hum, float hujan, float angin) {
  if (WiFi.status() != WL_CONNECTED) return;
  if (isnan(suhu) || isnan(hum) || isnan(hujan) || isnan(angin)) return;
  String ts = getTimestamp();
  if (ts == "1970-01-01 00:00:00") return;

  HTTPClient http;
  http.begin(clientSecure, scriptURL_historis);
  http.addHeader("Content-Type", "application/json");
  http.setUserAgent("ESP32-WeatherStation/2.0");
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  http.setRedirectLimit(3);
  http.setTimeout(8000);

  String payload = "{\"method\":\"append\",\"date\":\"" + ts + "\",";
  payload += "\"temperature\":" + String(suhu,1) + ",";
  payload += "\"humidity\":" + String(hum,1) + ",";
  payload += "\"rainfall\":" + String(hujan,1) + ",";
  payload += "\"windSpeed\":" + String(angin,1) + "}";

  int code = http.POST(payload);
  if (code == 200) {
    Serial.println("[HIST] OK");
  } else {
    Serial.printf("[HIST] HTTP %d\n", code);
    if (code == 400) {
      String resp = http.getString();
      Serial.println("Respon error: " + resp.substring(0, 150));
    }
  }
  http.end();
}

// ========== KIRIM PESAN TELEGRAM ==========
void sendTelegramMessage(String message) {
  if (WiFi.status() != WL_CONNECTED || !timeIsSynchronized) return;
  HTTPClient https;
  String encoded = message;
  encoded.replace(" ", "%20");
  encoded.replace("\n", "%0A");
  encoded.replace("/", "%2F");
  String url = "https://api.telegram.org/bot" + String(botToken) + "/sendMessage?chat_id=" + adminChatID + "&text=" + encoded;
  https.begin(clientSecure, url);
  https.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  int code = https.GET();
  if (code != 200) Serial.printf("Telegram error %d\n", code);
  https.end();
}

// ========== KIRIM PERINTAH PREDIKSI ==========
void kirimPerintahPrediksi(float suhu, float hum, float hujan, float angin) {
  if (WiFi.status() != WL_CONNECTED) return;
  String cmd = "/prediksi " + String(suhu,1) + " " + String(hum,1) + " " + String(hujan,1) + " " + String(angin,1);
  sendTelegramMessage(cmd);
  // Indikasi di LCD (opsional)
  if (lastWiFiStatus) {
    lcd.setCursor(15, 3);
    lcd.print("PRD");
    delay(500);
    lcd.setCursor(15, 3);
    lcd.print("   ");
  }
}

// ========== HANDLER UNTUK SERVER LOKAL (JSON) ==========
void handleSensorData() {
  String json = "{\"temperature\":" + String(suhu_terkalibrasi,1) +
                ",\"humidity\":" + String(hum_terkalibrasi,1) +
                ",\"rainfall\":" + String(curahHujan,1) +
                ",\"wind_speed\":" + String(kecepatanAngin,1) + "}";
  server.send(200, "application/json", json);
}