"""
RUGI-BATCH AI — Backend (FastAPI)
Deteksi kehilangan hasil produksi bawang merah goreng.

Endpoints:
  GET  /                  → status & kesiapan model
  POST /api/login         → autentikasi demo
  POST /api/analyze-batch → analisis satu batch
  POST /api/sync-offline  → sinkronisasi batch offline (array)

Prapemrosesan identik dengan train_model.py:
  RF  : [berat_awal_kg, suhu_goreng_c, durasi_menit,
         kode_A1, kode_A2, kode_A3, kode_B1, kode_B2, kode_B3]
  ISO : [rasio_tahap1, rasio_tahap2, rasio_tahap3, rasio_tahap4]
"""

from __future__ import annotations

from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
    email: str
    password: str


class BatchInput(BaseModel):
    berat_awal_kg:       float = Field(...,    gt=0, description="Berat awal bahan baku (kg)")
    harga_per_kg:        float = Field(42672.0, gt=0, description="Harga per kg (Rp)")
    kode_perlakuan:      str   = Field(...,          description="Kode perlakuan A1–B3")
    suhu_goreng_c:       float = Field(...,    gt=0, description="Suhu penggorengan (°C)")
    durasi_menit:        float = Field(...,    gt=0, description="Durasi penggorengan (menit)")
    berat_tahap1_kupas:  float = Field(...,    gt=0, description="Berat setelah pengupasan (kg)")
    berat_tahap2_iris:   float = Field(...,    gt=0, description="Berat setelah pengirisan + tepung (kg)")
    berat_tahap3_goreng: float = Field(...,    gt=0, description="Berat setelah penggorengan (kg)")
    berat_tahap4_kemas:  float = Field(...,    gt=0, description="Berat setelah pengemasan (kg)")


class SyncOfflineRequest(BaseModel):
    batches: List[BatchInput]


# ─────────────────────────────────────────────
# Helper: prapemrosesan identik dengan train_model.py
# ─────────────────────────────────────────────
def _build_rf_input(b: BatchInput) -> np.ndarray:
    """
    Vektor fitur RF:
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
    return np.array([b.berat_awal_kg, b.suhu_goreng_c, b.durasi_menit] + ohe,
                    dtype=float).reshape(1, -1)


def _build_iso_input(b: BatchInput) -> np.ndarray:
    """
    Vektor fitur IsoForest:
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
    """Jalankan analisis lengkap untuk satu BatchInput."""

    # Validasi model
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

    # 2. Prediksi rendemen dari RF
    prediksi_rendemen = round(float(rf_model.predict(_build_rf_input(b))[0]), 4)

    # 3. Selisih rendemen
    selisih_rendemen = round(prediksi_rendemen - rendemen_aktual, 4)

    # 4. Deteksi anomali
    pred         = iso_model.predict(_build_iso_input(b))[0]  # -1 anomali, 1 normal
    status_batch = "Anomali" if pred == -1 else "Normal"

    # 5. Cabang logika berdasarkan status_batch
    if status_batch == "Normal":
        # ── Batch Normal ──────────────────────────────────────────────
        tahap_bocor      = "Tidak Ada"
        rasio_bocor_pct  = 0.0
        estimasi_kerugian = 0.0
        rekomendasi = (
            "Produksi berjalan optimal sesuai standar rendemen. "
            "Pertahankan suhu dan durasi penggorengan saat ini."
        )

    else:
        # ── Batch Anomali ─────────────────────────────────────────────
        # Tahap dengan susut terbesar
        tahap_bocor, rasio_bocor_pct = _tahap_paling_bocor(b)

        # Estimasi kerugian:
        #   berat_produk_hilang   = selisih_rendemen/100 × berat_awal_kg
        #   bahan_baku_terbuang   = berat_produk_hilang / (prediksi_rendemen/100)
        #   estimasi_kerugian_Rp  = bahan_baku_terbuang × harga_per_kg
        if selisih_rendemen > 0 and prediksi_rendemen > 0:
            berat_produk_hilang = (selisih_rendemen / 100) * b.berat_awal_kg
            bahan_baku_terbuang = berat_produk_hilang / (prediksi_rendemen / 100)
            estimasi_kerugian   = round(bahan_baku_terbuang * b.harga_per_kg, 2)
        else:
            estimasi_kerugian = 0.0

        rekomendasi = REKOMENDASI_TAHAP.get(
            tahap_bocor,
            "Lakukan pemeriksaan menyeluruh pada semua tahap produksi.",
        )

    return {
        "rendemen_aktual_persen":          rendemen_aktual,
        "prediksi_rendemen_ai_persen":     prediksi_rendemen,
        "selisih_rendemen_persen":         selisih_rendemen,
        "status_batch":                    status_batch,
        "tahap_paling_bocor":              tahap_bocor,
        "rasio_susut_tahap_bocor_persen":  rasio_bocor_pct,
        "estimasi_kerugian_rupiah":        estimasi_kerugian,
        "rekomendasi_pemeriksaan":         rekomendasi,
    }


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", summary="Status API & kesiapan model")
def root():
    return {
        "status":    "aktif",
        "aplikasi":  "RUGI-BATCH AI",
        "versi":     "1.0.0",
        "model": {
            "rf_rendemen":  {"file": RF_PATH,  "siap": rf_model  is not None},
            "iso_anomali":  {"file": ISO_PATH, "siap": iso_model is not None},
        },
        "pesan": (
            "Semua model siap digunakan."
            if rf_model and iso_model
            else "Satu atau lebih model belum dimuat. Jalankan train_model.py terlebih dahulu."
        ),
    }


@app.post("/api/login", summary="Autentikasi pengguna demo")
def login(req: LoginRequest):
    if req.email != "demo@rugibatch.id" or req.password != "demo123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password salah.",
        )
    return {
        "status":       "berhasil",
        "access_token": "demo-token-rugibatch-2024",
        "token_type":   "bearer",
        "pengguna": {
            "email": "demo@rugibatch.id",
            "nama":  "Pengguna Demo",
            "peran": "operator",
        },
    }


@app.post("/api/analyze-batch", summary="Analisis satu batch produksi")
def analyze_batch(b: BatchInput):
    hasil = _analisis_batch(b)
    return {
        "status": "berhasil",
        "input":  b.model_dump(),
        "hasil":  hasil,
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
            hasil = _analisis_batch(b)
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
            "index":  idx,
            "status": "berhasil",
            "input":  b.model_dump(),
            "hasil":  hasil,
        })

    return {
        "status":                  "berhasil",
        "total_batch":             len(req.batches),
        "total_anomali":           total_anomali,
        "total_kerugian_rupiah":   round(total_kerugian, 2),
        "hasil_per_batch":         hasil_list,
    }
