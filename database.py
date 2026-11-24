"""
RUGI-BATCH AI — Database Layer
Menggunakan sqlite3 bawaan Python.

Tabel:
  users   — akun pengguna
  batches — riwayat pencatatan batch produksi
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "rugi_batch.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # akses kolom by name
    conn.execute("PRAGMA journal_mode=WAL")  # aman untuk concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Buat tabel jika belum ada dan seed akun demo."""
    conn = get_connection()
    cur  = conn.cursor()

    # ── Tabel users ───────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT    NOT NULL UNIQUE,
            nama       TEXT    NOT NULL,
            peran      TEXT    NOT NULL DEFAULT 'operator',
            password   TEXT    NOT NULL,
            dibuat_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)

    # ── Tabel batches ─────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            kode_batch                  TEXT    NOT NULL,
            tanggal                     TEXT    NOT NULL,
            user_id                     INTEGER REFERENCES users(id),

            -- Input produksi
            berat_awal_kg               REAL    NOT NULL,
            harga_per_kg                REAL    NOT NULL DEFAULT 42672,
            kode_perlakuan              TEXT    NOT NULL,
            suhu_goreng_c               REAL    NOT NULL,
            durasi_menit                REAL    NOT NULL,
            berat_tahap1_kupas          REAL    NOT NULL,
            berat_tahap2_iris           REAL    NOT NULL,
            berat_tahap3_goreng         REAL    NOT NULL,
            berat_tahap4_kemas          REAL    NOT NULL,

            -- Hasil analisis AI
            rendemen_aktual_persen      REAL,
            prediksi_rendemen_normal    REAL,
            selisih_rendemen_persen     REAL,
            status_batch                TEXT,          -- Normal | Bocor | Waspada
            tahap_paling_bocor          TEXT,
            rasio_susut_tahap_bocor_persen REAL,
            estimasi_kerugian_rupiah    REAL,
            berat_hilang_kg             REAL,
            rekomendasi_pemeriksaan     TEXT,

            dibuat_at                   TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )
    """)

    # ── Seed akun demo (hanya jika belum ada) ─────────────────────────
    cur.execute("SELECT id FROM users WHERE email = ?", ("sari@bawanghangat.com",))
    if cur.fetchone() is None:
        cur.execute("""
            INSERT INTO users (email, nama, peran, password)
            VALUES (?, ?, ?, ?)
        """, ("sari@bawanghangat.com", "Bu Sari", "pemilik", "demo123"))

    # Akun lama (tetap didukung untuk kompatibilitas)
    cur.execute("SELECT id FROM users WHERE email = ?", ("demo@rugibatch.id",))
    if cur.fetchone() is None:
        cur.execute("""
            INSERT INTO users (email, nama, peran, password)
            VALUES (?, ?, ?, ?)
        """, ("demo@rugibatch.id", "Pengguna Demo", "operator", "demo123"))

    conn.commit()
    conn.close()
    print(f"[DB] Database '{DB_PATH}' siap.")


def next_kode_batch(conn: sqlite3.Connection) -> str:
    """Generate kode batch berikutnya, misal BG-015."""
    row = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS total FROM batches"
    ).fetchone()
    nomor = (row["total"] or 0) + 1
    return f"BG-{nomor:03d}"


def insert_batch(conn: sqlite3.Connection, data: dict) -> int:
    """Simpan satu baris batch, kembalikan id."""
    data.setdefault("dibuat_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cur = conn.execute("""
        INSERT INTO batches (
            kode_batch, tanggal, user_id,
            berat_awal_kg, harga_per_kg, kode_perlakuan,
            suhu_goreng_c, durasi_menit,
            berat_tahap1_kupas, berat_tahap2_iris,
            berat_tahap3_goreng, berat_tahap4_kemas,
            rendemen_aktual_persen, prediksi_rendemen_normal,
            selisih_rendemen_persen, status_batch,
            tahap_paling_bocor, rasio_susut_tahap_bocor_persen,
            estimasi_kerugian_rupiah, berat_hilang_kg,
            rekomendasi_pemeriksaan, dibuat_at
        ) VALUES (
            :kode_batch, :tanggal, :user_id,
            :berat_awal_kg, :harga_per_kg, :kode_perlakuan,
            :suhu_goreng_c, :durasi_menit,
            :berat_tahap1_kupas, :berat_tahap2_iris,
            :berat_tahap3_goreng, :berat_tahap4_kemas,
            :rendemen_aktual_persen, :prediksi_rendemen_normal,
            :selisih_rendemen_persen, :status_batch,
            :tahap_paling_bocor, :rasio_susut_tahap_bocor_persen,
            :estimasi_kerugian_rupiah, :berat_hilang_kg,
            :rekomendasi_pemeriksaan, :dibuat_at
        )
    """, data)
    conn.commit()
    return cur.lastrowid


def fetch_batches(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """Ambil riwayat batch terbaru, diurutkan dari yang terbaru."""
    rows = conn.execute("""
        SELECT * FROM batches
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def fetch_dashboard_summary(conn: sqlite3.Connection) -> dict:
    """
    Hitung statistik agregat untuk layar beranda.
    Periode: seluruh data yang ada di DB.
    """
    total_row = conn.execute(
        "SELECT COUNT(*) AS total FROM batches"
    ).fetchone()
    total_batch = total_row["total"] or 0

    if total_batch == 0:
        return {
            "total_batch":           0,
            "batch_bermasalah":      0,
            "rendemen_rata_persen":  0.0,
            "total_kerugian_rupiah": 0.0,
            "tahap_paling_boros":    "-",
            "riwayat_grafik":        [],
        }

    agg = conn.execute("""
        SELECT
            COUNT(*)                                       AS total,
            SUM(CASE WHEN status_batch != 'Normal' THEN 1 ELSE 0 END) AS bermasalah,
            AVG(rendemen_aktual_persen)                    AS avg_rendemen,
            SUM(estimasi_kerugian_rupiah)                  AS total_kerugian
        FROM batches
    """).fetchone()

    # Tahap paling sering menjadi titik bocor
    tahap_row = conn.execute("""
        SELECT tahap_paling_bocor, COUNT(*) AS cnt
        FROM batches
        WHERE status_batch != 'Normal'
          AND tahap_paling_bocor IS NOT NULL
          AND tahap_paling_bocor != 'Tidak Ada'
        GROUP BY tahap_paling_bocor
        ORDER BY cnt DESC
        LIMIT 1
    """).fetchone()
    tahap_boros = tahap_row["tahap_paling_bocor"] if tahap_row else "-"

    # Riwayat 12 batch terakhir untuk grafik
    rows_grafik = conn.execute("""
        SELECT kode_batch, rendemen_aktual_persen, status_batch
        FROM batches
        ORDER BY id DESC
        LIMIT 12
    """).fetchall()
    grafik = [
        {
            "kode_batch":            r["kode_batch"],
            "rendemen_aktual_persen": round(r["rendemen_aktual_persen"] or 0, 2),
            "status_batch":          r["status_batch"],
        }
        for r in reversed(rows_grafik)   # urutan kronologis
    ]

    return {
        "total_batch":           total_batch,
        "batch_bermasalah":      agg["bermasalah"] or 0,
        "rendemen_rata_persen":  round(agg["avg_rendemen"] or 0, 2),
        "total_kerugian_rupiah": round(agg["total_kerugian"] or 0, 2),
        "tahap_paling_boros":    tahap_boros,
        "riwayat_grafik":        grafik,
    }
