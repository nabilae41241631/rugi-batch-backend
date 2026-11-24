"""
RUGI-BATCH AI — Backend (FastAPI)  rev-3
Deteksi kehilangan hasil produksi bawang merah goreng.

Endpoints:
  GET  /                      → status & kesiapan model
  POST /api/login             → autentikasi demo (sari@bawanghangat.com / demo@rugibatch.id)
  POST /api/analyze-batch     → analisis satu batch + simpan ke DB
  POST /api/sync-offline      → sinkronisasi batch offline (array)
  GET  /api/dashboard-summary → statistik agregat untuk layar Beranda
  GET  /api/batches           → riwayat batch untuk layar Riwayat & Pemeriksaan

Prapemrosesan identik dengan train_model.py:
  RF  : [berat_awal_kg, suhu_goreng_c, durasi_menit,
         kode_A1, kode_A2, kode_A3, kode_B1, kode_B2, kode_B3]
  ISO : [rasio_tahap1, rasio_tahap2, rasio_tahap3, rasio_tahap4]
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database as db

# ─────────────────────────────────────────────
# Konstanta (harus sama dengan train_model.py)
# ─────────────────────────────────────────────
KODE_LIST = ["A1", "A2", "A3", "B1", "B2", "B3"]

REKOMENDASI_TAHAP = {
    "Tahap 1 - Kupas": (
        "Periksa ketajaman pisau kupas dan kecepatan operator. "
        "Susut kupas melebihi batas normal 12–15 %. "
        "Lakukan kalibrasi ulang proses pengupasan."
    ),
    "Tahap 2 - Iris": (
        "Periksa rasio penambahan tepung sesuai kode perlakuan. "
        "Pastikan timbangan tepung terkalibrasi dan tidak terjadi tumpah."
    ),
    "Tahap 3 - Goreng": (
        "Periksa suhu dan durasi penggorengan. "
        "Suhu terlalu tinggi atau durasi terlalu lama menyebabkan susut berlebih. "
        "Sesuaikan dengan parameter optimal kode perlakuan yang digunakan."
    ),
    "Tahap 4 - Kemas": (
        "Periksa proses pengemasan dan timbangan output. "
        "Pastikan tidak ada produk jatuh atau tertinggal di wadah goreng."
    ),
}

# Akun demo yang didukung (email → (nama, peran))
DEMO_ACCOUNTS = {
    "sari@bawanghangat.com": ("Bu Sari",       "pemilik"),
    "demo@rugibatch.id":     ("Pengguna Demo",  "operator"),
}
DEMO_PASSWORD = "demo123"

# ─────────────────────────────────────────────
# Muat model saat startup
# ─────────────────────────────────────────────
RF_PATH  = "rf_rendemen.joblib"
ISO_PATH = "iso_anomali.joblib"

rf_model  = None
iso_model = None

try:
    rf_model = joblib.load(RF_PATH)
except FileNotFoundError:
    pass

try:
    iso_model = joblib.load(ISO_PATH)
except FileNotFoundError:
    pass

# Inisialisasi database (buat tabel & seed akun demo)
db.init_db()


# ─────────────────────────────────────────────
# Aplikasi FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="RUGI-BATCH AI",
    description="Backend deteksi kehilangan hasil produksi bawang merah goreng.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schema Pydantic
# ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    email:    str
    password: str


class BatchInput(BaseModel):
    berat_awal_kg:       float = Field(...,     gt=0, description="Berat awal bahan baku (kg)")
    harga_per_kg:        float = Field(42672.0, gt=0, description="Harga per kg (Rp)")
    kode_perlakuan:      str   = Field(...,           description="Kode perlakuan A1–B3")
    suhu_goreng_c:       float = Field(...,     gt=0, description="Suhu penggorengan (°C)")
    durasi_menit:        float = Field(...,     gt=0, description="Durasi penggorengan (menit)")
    berat_tahap1_kupas:  float = Field(...,     gt=0, description="Berat setelah pengupasan (kg)")
    berat_tahap2_iris:   float = Field(...,     gt=0, description="Berat setelah pengirisan + tepung (kg)")
    berat_tahap3_goreng: float = Field(...,     gt=0, description="Berat setelah penggorengan (kg)")
    berat_tahap4_kemas:  float = Field(...,     gt=0, description="Berat setelah pengemasan (kg)")


class SyncOfflineRequest(BaseModel):
    batches: List[BatchInput]


# ─────────────────────────────────────────────
# Helper: prapemrosesan identik dengan train_model.py
# ─────────────────────────────────────────────
def _build_rf_input(b: BatchInput) -> np.ndarray:
    """
    Vektor fitur RF (9 kolom):
    [berat_awal_kg, suhu_goreng_c, durasi_menit,
     kode_A1, kode_A2, kode_A3, kode_B1, kode_B2, kode_B3]
    """
    kode = b.kode_perlakuan.upper()
    if kode not in KODE_LIST:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"kode_perlakuan '{kode}' tidak valid. Pilihan: {KODE_LIST}",
        )
    ohe = [1.0 if kode == k else 0.0 for k in KODE_LIST]
    return np.array(
        [b.berat_awal_kg, b.suhu_goreng_c, b.durasi_menit] + ohe,
        dtype=float,
    ).reshape(1, -1)


def _build_iso_input(b: BatchInput) -> np.ndarray:
    """
    Vektor fitur IsoForest (4 kolom):
    [rasio_tahap1, rasio_tahap2, rasio_tahap3, rasio_tahap4]
    rasio = 1 - (berat_keluar / berat_masuk)
    """
    r1 = 1 - (b.berat_tahap1_kupas  / b.berat_awal_kg)
    r2 = 1 - (b.berat_tahap2_iris   / b.berat_tahap1_kupas)
    r3 = 1 - (b.berat_tahap3_goreng / b.berat_tahap2_iris)
    r4 = 1 - (b.berat_tahap4_kemas  / b.berat_tahap3_goreng)
    return np.array([r1, r2, r3, r4], dtype=float).reshape(1, -1)


def _tahap_paling_bocor(b: BatchInput) -> tuple[str, float]:
    """Kembalikan (nama_tahap, persen_susut) dengan susut terbesar."""
    tahap = {
        "Tahap 1 - Kupas":  1 - (b.berat_tahap1_kupas  / b.berat_awal_kg),
        "Tahap 2 - Iris":   1 - (b.berat_tahap2_iris   / b.berat_tahap1_kupas),
        "Tahap 3 - Goreng": 1 - (b.berat_tahap3_goreng / b.berat_tahap2_iris),
        "Tahap 4 - Kemas":  1 - (b.berat_tahap4_kemas  / b.berat_tahap3_goreng),
    }
    nama = max(tahap, key=lambda k: tahap[k])
    return nama, round(tahap[nama] * 100, 2)


def _analisis_batch(b: BatchInput) -> dict:
    """
    Jalankan analisis lengkap untuk satu BatchInput.
    Kembalikan dict hasil yang siap disimpan ke DB maupun dikembalikan ke klien.
    """
    if rf_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model rf_rendemen.joblib belum dimuat. Jalankan train_model.py.",
        )
    if iso_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model iso_anomali.joblib belum dimuat. Jalankan train_model.py.",
        )

    # 1. Rendemen aktual
    rendemen_aktual = round((b.berat_tahap4_kemas / b.berat_awal_kg) * 100, 4)

    # 2. Prediksi rendemen normal dari RF
    prediksi_rendemen = round(float(rf_model.predict(_build_rf_input(b))[0]), 4)

    # 3. Selisih rendemen (positif = rendemen aktual lebih rendah dari prediksi)
    selisih_rendemen = round(prediksi_rendemen - rendemen_aktual, 4)

    # 4. Deteksi anomali via IsolationForest
    pred         = iso_model.predict(_build_iso_input(b))[0]  # -1 anomali, 1 normal
    status_batch = "Anomali" if pred == -1 else "Normal"

    # 5. Cabang logika berdasarkan status_batch
    if status_batch == "Normal":
        # ── Normal ───────────────────────────────────────────────────
        tahap_bocor       = "Tidak Ada"
        rasio_bocor_pct   = 0.0
        estimasi_kerugian = 0.0
        berat_hilang      = 0.0
        rekomendasi = (
            "Produksi berjalan optimal sesuai standar rendemen. "
            "Pertahankan suhu dan durasi penggorengan saat ini."
        )

    else:
        # ── Anomali ───────────────────────────────────────────────────
        tahap_bocor, rasio_bocor_pct = _tahap_paling_bocor(b)

        # Estimasi kerugian:
        #   berat_produk_hilang = selisih_rendemen/100 × berat_awal_kg
        #   bahan_baku_terbuang = berat_produk_hilang / (prediksi_rendemen/100)
        #   kerugian_Rp         = bahan_baku_terbuang × harga_per_kg
        if selisih_rendemen > 0 and prediksi_rendemen > 0:
            berat_produk_hilang = (selisih_rendemen / 100) * b.berat_awal_kg
            bahan_baku_terbuang = berat_produk_hilang / (prediksi_rendemen / 100)
            estimasi_kerugian   = round(bahan_baku_terbuang * b.harga_per_kg, 2)
            berat_hilang        = round(berat_produk_hilang, 4)
        else:
            estimasi_kerugian = 0.0
            berat_hilang      = 0.0

        rekomendasi = REKOMENDASI_TAHAP.get(
            tahap_bocor,
            "Lakukan pemeriksaan menyeluruh pada semua tahap produksi.",
        )

    return {
        # ── Kolom hasil analisis (cocok dengan schema DB) ──
        "rendemen_aktual_persen":          rendemen_aktual,
        "prediksi_rendemen_normal":        prediksi_rendemen,   # nama kolom DB
        "selisih_rendemen_persen":         selisih_rendemen,
        "status_batch":                    status_batch,
        "tahap_paling_bocor":              tahap_bocor,
        "rasio_susut_tahap_bocor_persen":  rasio_bocor_pct,
        "estimasi_kerugian_rupiah":        estimasi_kerugian,
        "berat_hilang_kg":                 berat_hilang,
        "rekomendasi_pemeriksaan":         rekomendasi,
        # ── Alias tampilan HTML (prediksi_rendemen_ai_persen) ──
        "prediksi_rendemen_ai_persen":     prediksi_rendemen,
    }


def _simpan_ke_db(b: BatchInput, hasil: dict, user_id: Optional[int] = None) -> str:
    """Simpan satu hasil analisis ke rugi_batch.db, kembalikan kode_batch."""
    conn = db.get_connection()
    try:
        kode_batch = db.next_kode_batch(conn)
        tanggal    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        db.insert_batch(conn, {
            "kode_batch":                   kode_batch,
            "tanggal":                      tanggal,
            "user_id":                      user_id,
            # input produksi
            "berat_awal_kg":                b.berat_awal_kg,
            "harga_per_kg":                 b.harga_per_kg,
            "kode_perlakuan":               b.kode_perlakuan.upper(),
            "suhu_goreng_c":                b.suhu_goreng_c,
            "durasi_menit":                 b.durasi_menit,
            "berat_tahap1_kupas":           b.berat_tahap1_kupas,
            "berat_tahap2_iris":            b.berat_tahap2_iris,
            "berat_tahap3_goreng":          b.berat_tahap3_goreng,
            "berat_tahap4_kemas":           b.berat_tahap4_kemas,
            # hasil analisis
            "rendemen_aktual_persen":       hasil["rendemen_aktual_persen"],
            "prediksi_rendemen_normal":     hasil["prediksi_rendemen_normal"],
            "selisih_rendemen_persen":      hasil["selisih_rendemen_persen"],
            "status_batch":                 hasil["status_batch"],
            "tahap_paling_bocor":           hasil["tahap_paling_bocor"],
            "rasio_susut_tahap_bocor_persen": hasil["rasio_susut_tahap_bocor_persen"],
            "estimasi_kerugian_rupiah":     hasil["estimasi_kerugian_rupiah"],
            "berat_hilang_kg":              hasil["berat_hilang_kg"],
            "rekomendasi_pemeriksaan":      hasil["rekomendasi_pemeriksaan"],
        })
        return kode_batch
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", summary="Status API & kesiapan model")
def root():
    return {
        "status":   "aktif",
        "aplikasi": "RUGI-BATCH AI",
        "versi":    "1.0.0",
        "model": {
            "rf_rendemen": {"file": RF_PATH,  "siap": rf_model  is not None},
            "iso_anomali": {"file": ISO_PATH, "siap": iso_model is not None},
        },
        "pesan": (
            "Semua model siap digunakan."
            if rf_model and iso_model
            else "Satu atau lebih model belum dimuat. Jalankan train_model.py terlebih dahulu."
        ),
    }


@app.post("/api/login", summary="Autentikasi pengguna demo")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    if email not in DEMO_ACCOUNTS or req.password != DEMO_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah.",
        )
    nama, peran = DEMO_ACCOUNTS[email]
    return {
        "status":       "berhasil",
        "access_token": f"demo-token-rugibatch-{email.split('@')[0]}",
        "token_type":   "bearer",
        "pengguna": {
            "email": email,
            "nama":  nama,
            "peran": peran,
        },
    }


@app.post("/api/analyze-batch", summary="Analisis satu batch produksi dan simpan ke DB")
def analyze_batch(b: BatchInput):
    hasil      = _analisis_batch(b)
    kode_batch = _simpan_ke_db(b, hasil)

    return {
        "status":     "berhasil",
        "kode_batch": kode_batch,
        "input":      b.model_dump(),
        "hasil": {
            # Nama field yang dipakai di layar HTML
            "rendemen_aktual_persen":         hasil["rendemen_aktual_persen"],
            "prediksi_rendemen_ai_persen":    hasil["prediksi_rendemen_ai_persen"],
            "selisih_rendemen_persen":        hasil["selisih_rendemen_persen"],
            "status_batch":                   hasil["status_batch"],
            "tahap_paling_bocor":             hasil["tahap_paling_bocor"],
            "rasio_susut_tahap_bocor_persen": hasil["rasio_susut_tahap_bocor_persen"],
            "estimasi_kerugian_rupiah":       hasil["estimasi_kerugian_rupiah"],
            "berat_hilang_kg":                hasil["berat_hilang_kg"],
            "rekomendasi_pemeriksaan":        hasil["rekomendasi_pemeriksaan"],
        },
    }


@app.post("/api/sync-offline", summary="Sinkronisasi batch offline (array)")
def sync_offline(req: SyncOfflineRequest):
    if not req.batches:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Array 'batches' kosong. Kirim minimal satu data batch.",
        )

    hasil_list     = []
    total_anomali  = 0
    total_kerugian = 0.0

    for idx, b in enumerate(req.batches):
        try:
            hasil      = _analisis_batch(b)
            kode_batch = _simpan_ke_db(b, hasil)
        except HTTPException as exc:
            hasil_list.append({
                "index":  idx,
                "status": "error",
                "detail": exc.detail,
                "input":  b.model_dump(),
            })
            continue

        if hasil["status_batch"] == "Anomali":
            total_anomali += 1
        total_kerugian += hasil["estimasi_kerugian_rupiah"]

        hasil_list.append({
            "index":      idx,
            "status":     "berhasil",
            "kode_batch": kode_batch,
            "input":      b.model_dump(),
            "hasil": {
                "rendemen_aktual_persen":         hasil["rendemen_aktual_persen"],
                "prediksi_rendemen_ai_persen":    hasil["prediksi_rendemen_ai_persen"],
                "selisih_rendemen_persen":        hasil["selisih_rendemen_persen"],
                "status_batch":                   hasil["status_batch"],
                "tahap_paling_bocor":             hasil["tahap_paling_bocor"],
                "rasio_susut_tahap_bocor_persen": hasil["rasio_susut_tahap_bocor_persen"],
                "estimasi_kerugian_rupiah":       hasil["estimasi_kerugian_rupiah"],
                "berat_hilang_kg":                hasil["berat_hilang_kg"],
                "rekomendasi_pemeriksaan":        hasil["rekomendasi_pemeriksaan"],
            },
        })

    return {
        "status":                "berhasil",
        "total_batch":           len(req.batches),
        "total_anomali":         total_anomali,
        "total_kerugian_rupiah": round(total_kerugian, 2),
        "hasil_per_batch":       hasil_list,
    }


@app.get("/api/dashboard-summary", summary="Statistik agregat untuk layar Beranda")
def dashboard_summary():
    """
    Mengembalikan data yang dibutuhkan layar Beranda (layar_2_beranda):
    - total_batch, batch_bermasalah, rendemen_rata_persen
    - total_kerugian_rupiah, tahap_paling_boros
    - riwayat_grafik: 12 batch terakhir (kode_batch, rendemen, status)
    - alert_mingguan: batch bermasalah + total kerugian 7 hari terakhir
    """
    conn = db.get_connection()
    try:
        summary = db.fetch_dashboard_summary(conn)

        # Tambah data alert mingguan (7 hari terakhir) untuk banner merah di beranda
        weekly = conn.execute("""
            SELECT
                SUM(CASE WHEN status_batch != 'Normal' THEN 1 ELSE 0 END) AS bocor,
                SUM(estimasi_kerugian_rupiah) AS rugi
            FROM batches
            WHERE tanggal >= datetime('now', '-7 days', 'localtime')
        """).fetchone()

        summary["alert_mingguan"] = {
            "batch_bocor_7_hari":    (weekly["bocor"] or 0) if weekly else 0,
            "total_kerugian_7_hari": round((weekly["rugi"] or 0.0) if weekly else 0.0, 2),
        }

        return {"status": "berhasil", "data": summary}
    finally:
        conn.close()


@app.get("/api/batches", summary="Riwayat batch untuk layar Riwayat & Pemeriksaan")
def get_batches(
    limit:  int = Query(default=50,  ge=1, le=200, description="Jumlah batch yang diambil"),
    offset: int = Query(default=0,   ge=0,          description="Lewati N baris pertama"),
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter: Normal | Anomali | semua",
    ),
):
    """
    Kembalikan daftar batch sesuai kebutuhan layar Riwayat & Pemeriksaan
    (layar_4_riwayat_dan_hasil_pemeriksaan).

    Setiap item memuat field yang ditampilkan di kartu riwayat:
    kode_batch, tanggal, rendemen_aktual_persen, status_batch,
    estimasi_kerugian_rupiah, tahap_paling_bocor, rekomendasi_pemeriksaan,
    ditambah seluruh data input & analisis lengkap.
    """
    conn = db.get_connection()
    try:
        query  = "SELECT * FROM batches"
        params: list = []

        if status_filter and status_filter.lower() != "semua":
            query  += " WHERE status_batch = ?"
            params.append(status_filter)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        rows = conn.execute(query, params).fetchall()

        # Hitung total untuk paginasi
        count_query = "SELECT COUNT(*) AS total FROM batches"
        count_params: list = []
        if status_filter and status_filter.lower() != "semua":
            count_query  += " WHERE status_batch = ?"
            count_params.append(status_filter)
        total = conn.execute(count_query, count_params).fetchone()["total"]

        batches = []
        for r in rows:
            row = dict(r)
            # Alias untuk kompatibilitas tampilan HTML
            row["prediksi_rendemen_ai_persen"] = row.get("prediksi_rendemen_normal")
            batches.append(row)

        return {
            "status":       "berhasil",
            "total":        total,
            "limit":        limit,
            "offset":       offset,
            "batches":      batches,
        }
    finally:
        conn.close()
