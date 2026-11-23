"""
RUGI-BATCH AI — Train Model  (rev-2)
Deteksi kehilangan hasil produksi bawang merah goreng.

Pipeline:
  1. Bangkitkan 400 baris data simulasi terkalibrasi → dataset_bawang_goreng.csv
  2. Latih RandomForestRegressor (hanya baris normal) → rf_rendemen.joblib
  3. Latih IsolationForest (seluruh 400 baris)        → iso_anomali.joblib

Referensi rendemen normal: Anggarkasih dkk.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ─────────────────────────────────────────────
# 0. Seed global
# ─────────────────────────────────────────────
RNG = np.random.default_rng(42)
N   = 400

# ─────────────────────────────────────────────
# 1. Parameter rendemen target per kode perlakuan
#    (Anggarkasih dkk.)
# ─────────────────────────────────────────────
RENDEMEN_TARGET = {
    "A1": 31.38,
    "A2": 32.25,
    "A3": 37.33,
    "B1": 25.83,
    "B2": 29.50,
    "B3": 29.50,
}

# Faktor penambahan bobot tepung per kode perlakuan
TEPUNG_RASIO = {
    "A1": 0.00,
    "A2": 0.05,
    "A3": 0.10,
    "B1": 0.00,
    "B2": 0.05,
    "B3": 0.10,
}

# Suhu optimum per kode perlakuan (°C) — dasar kausalitas simulasi
SUHU_OPTIMAL = {
    "A1": 160.0,
    "A2": 165.0,
    "A3": 170.0,
    "B1": 155.0,
    "B2": 160.0,
    "B3": 165.0,
}

# Durasi optimum per kode perlakuan (menit)
DURASI_OPTIMAL = {
    "A1": 10.5,
    "A2": 10.8,
    "A3": 11.0,
    "B1": 10.0,
    "B2": 10.5,
    "B3": 10.5,
}

KODE_LIST = list(RENDEMEN_TARGET.keys())

# ─────────────────────────────────────────────
# 2. Bangkitkan fitur dasar
# ─────────────────────────────────────────────
berat_awal   = RNG.uniform(50,  200, N)
harga_per_kg = np.full(N, 42_672.0)
kode         = RNG.choice(KODE_LIST, N)
suhu_goreng  = RNG.uniform(130, 190, N)
durasi_menit = RNG.uniform(8.85, 12.0, N)

# ─────────────────────────────────────────────
# 3. Flag anomali (tepat 15 % = 60 baris)
# ─────────────────────────────────────────────
n_anomali   = int(N * 0.15)          # 60 baris
idx_anomali = RNG.choice(N, size=n_anomali, replace=False)
is_anomali  = np.zeros(N, dtype=bool)
is_anomali[idx_anomali] = True

jenis_anomali = RNG.integers(0, 2, N)       # 0 = kupas, 1 = goreng
susut_ekstra  = RNG.uniform(0.05, 0.12, N)  # 5–12 % ekstra

# ─────────────────────────────────────────────
# 4. Hitung berat tiap tahap + rendemen
#
#  Rendemen normal dibuat kausal terhadap suhu & durasi:
#    rendemen = rendemen_target
#               + alfa * (suhu - suhu_optimal)        ← pengaruh suhu
#               + beta * (durasi - durasi_optimal)    ← pengaruh durasi
#               + noise kecil
#
#  alfa dan beta dikalibrasi agar variasi total ~±2 %,
#  sehingga RF punya sinyal yang cukup namun R² tetap tinggi.
# ─────────────────────────────────────────────
ALFA =  0.08   # % rendemen per °C di atas/bawah optimum
BETA =  0.60   # % rendemen per menit di atas/bawah optimum
NOISE_STD = 0.25  # noise acak (standar deviasi %)

berat_t1 = np.zeros(N)
berat_t2 = np.zeros(N)
berat_t3 = np.zeros(N)
berat_t4 = np.zeros(N)
rendemen  = np.zeros(N)

for i in range(N):
    kd = kode[i]

    # ── Tahap 1: Kupas  (susut normal 12–15 %)
    susut_kupas = RNG.uniform(0.12, 0.15)
    if is_anomali[i] and jenis_anomali[i] == 0:
        susut_kupas += susut_ekstra[i]      # anomali di kupas
    t1 = berat_awal[i] * (1 - susut_kupas)
    berat_t1[i] = t1

    # ── Tahap 2: Iris + tambah tepung
    t2 = t1 * (1 + TEPUNG_RASIO[kd])
    berat_t2[i] = t2

    # ── Tahap 3: Goreng — rendemen kausal
    delta_suhu   = suhu_goreng[i]  - SUHU_OPTIMAL[kd]
    delta_durasi = durasi_menit[i] - DURASI_OPTIMAL[kd]

    rendemen_nominal = (
        RENDEMEN_TARGET[kd]
        + ALFA * delta_suhu
        + BETA * delta_durasi
        + RNG.normal(0, NOISE_STD)
    )

    if is_anomali[i] and jenis_anomali[i] == 1:
        rendemen_nominal -= susut_ekstra[i] * 100   # anomali di goreng

    rendemen_nominal = max(rendemen_nominal, 5.0)   # floor

    t3 = berat_awal[i] * rendemen_nominal / 100
    berat_t3[i] = t3
    rendemen[i] = rendemen_nominal

    # ── Tahap 4: Kemas (susut 1–3 %)
    susut_kemas  = RNG.uniform(0.01, 0.03)
    berat_t4[i]  = t3 * (1 - susut_kemas)

# ─────────────────────────────────────────────
# 5. Rakit DataFrame dan simpan CSV
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "berat_awal_kg":       berat_awal,
    "harga_per_kg":        harga_per_kg,
    "kode_perlakuan":      kode,
    "suhu_goreng_c":       suhu_goreng,
    "durasi_menit":        durasi_menit,
    "berat_tahap1_kupas":  berat_t1,
    "berat_tahap2_iris":   berat_t2,
    "berat_tahap3_goreng": berat_t3,
    "berat_tahap4_kemas":  berat_t4,
    "rendemen_persen":     rendemen,
    "is_anomali":          is_anomali.astype(int),
})

CSV_PATH = "dataset_bawang_goreng.csv"
df.to_csv(CSV_PATH, index=False)
print(f"[INFO] Dataset disimpan → {CSV_PATH}  ({len(df)} baris)")
print(f"[INFO] Baris normal : {(df['is_anomali'] == 0).sum()}")
print(f"[INFO] Baris anomali: {(df['is_anomali'] == 1).sum()}")

# ─────────────────────────────────────────────
# 6. RandomForestRegressor — hanya baris NORMAL
#    Fitur: berat_awal_kg, kode_perlakuan (OHE),
#           suhu_goreng_c, durasi_menit
#    Target: rendemen_persen
# ─────────────────────────────────────────────
df_normal = df[df["is_anomali"] == 0].copy().reset_index(drop=True)
print(f"\n[INFO] Data normal untuk RF: {len(df_normal)} baris")

# One-Hot Encoding kode_perlakuan
df_ohe = pd.get_dummies(df_normal["kode_perlakuan"], prefix="kode")

# Pastikan semua kolom kode hadir (penting saat inferensi nanti)
for kd in KODE_LIST:
    col = f"kode_{kd}"
    if col not in df_ohe.columns:
        df_ohe[col] = 0
df_ohe = df_ohe[[f"kode_{kd}" for kd in KODE_LIST]]

FITUR_NUMERIK = ["berat_awal_kg", "suhu_goreng_c", "durasi_menit"]
X_num = df_normal[FITUR_NUMERIK].values
X_ohe = df_ohe.values.astype(float)
X_rf  = np.hstack([X_num, X_ohe])
y_rf  = df_normal["rendemen_persen"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n[RandomForestRegressor — rendemen_persen (data normal saja)]")
print(f"  Jumlah train : {len(X_train)} baris")
print(f"  Jumlah test  : {len(X_test)} baris")
print(f"  R²           : {r2:.4f}")
print(f"  MAE          : {mae:.4f} %")

RF_PATH = "rf_rendemen.joblib"
joblib.dump(rf, RF_PATH)
print(f"  Model disimpan → {RF_PATH}")

# ─────────────────────────────────────────────
# 7. IsolationForest — seluruh 400 baris
#    Fitur: rasio penyusutan antar tahap
#      rasio_tahap1 = susut kupas
#      rasio_tahap2 = susut iris (negatif jika tepung ditambah)
#      rasio_tahap3 = susut goreng
#      rasio_tahap4 = susut kemas
# ─────────────────────────────────────────────
df["rasio_tahap1"] = 1 - (df["berat_tahap1_kupas"]  / df["berat_awal_kg"])
df["rasio_tahap2"] = 1 - (df["berat_tahap2_iris"]   / df["berat_tahap1_kupas"])
df["rasio_tahap3"] = 1 - (df["berat_tahap3_goreng"] / df["berat_tahap2_iris"])
df["rasio_tahap4"] = 1 - (df["berat_tahap4_kemas"]  / df["berat_tahap3_goreng"])

FITUR_ISO = ["rasio_tahap1", "rasio_tahap2", "rasio_tahap3", "rasio_tahap4"]
X_iso = df[FITUR_ISO].values

iso = IsolationForest(contamination=0.15, random_state=42)
iso.fit(X_iso)

pred_iso     = iso.predict(X_iso)       # -1 = anomali, 1 = normal
n_terdeteksi = (pred_iso == -1).sum()

print(f"\n[IsolationForest — deteksi anomali batch (seluruh 400 baris)]")
print(f"  Batch berlabel anomali (ground truth) : {is_anomali.sum()}")
print(f"  Batch terdeteksi anomali oleh model   : {n_terdeteksi}")

ISO_PATH = "iso_anomali.joblib"
joblib.dump(iso, ISO_PATH)
print(f"  Model disimpan → {ISO_PATH}")

print("\n[SELESAI] Semua model dan dataset berhasil dibuat.")
