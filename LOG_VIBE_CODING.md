# LOG VIBE CODING - RUGI-BATCH AI (TIM CEPLES)
- Anggota Pengembang Backend & AI: Nabila Istiya Arum Zahfa
- Perangkat Lunak: Visual Studio Code & Kiro AI Assistant
- Bahasa & Pustaka: Python, Pandas, NumPy, Scikit-Learn, Joblib, FastAPI, Uvicorn

---

## Sesi 1: Pembuatan Dataset Sintetis Terkalibrasi dan Pelatihan Awal Model AI
- Tanggal: 23 November 2026
- Berkas Target: `train_model.py`
- Prompt Utama ke Kiro:
  > "Buatkan kode Python lengkap pada file train_model.py untuk sistem RUGI-BATCH AI deteksi kehilangan hasil produksi bawang merah goreng. Bangkitkan 400 baris data simulasi terkalibrasi (random_state=42) dengan fitur berat_awal_kg, harga_per_kg (default 42672), kode_perlakuan (A1-B3), suhu_goreng_c (130-190), durasi_menit (8.85-12.0), serta berat 4 tahapan produksi. Kalibrasi rendemen normal sesuai jurnal Anggarkasih dkk. (25.83% - 37.33%), sisipkan 15% data anomali, latih RandomForestRegressor dan IsolationForest, lalu simpan model ke rf_rendemen.joblib dan iso_anomali.joblib."
- Audit & Intervensi Manual Manusia:
  1. Menolak perintah instalasi pustaka global otomatis dari Kiro dan mengarahkan eksekusi skrip ke lingkungan virtual terisolasi (`venv`).
  2. Mengevaluasi hasil pelatihan pertama dan menemukan kelemahan metodologis: nilai akurasi `R²` RandomForestRegressor hanya mencapai `0.4106` (MAE `1.8322%`) karena Kiro melatih model regresi menggunakan seluruh 400 baris data yang masih tercampur dengan 60 baris data anomali.

## Sesi 2: Perbaikan Metodologi Pemisahan Data Latih (Data Leakage & Anomaly Isolation)
- Tanggal: 23 November 2026
- Berkas Target: `train_model.py`
- Prompt Perbaikan ke Kiro:
  > "Perbaiki skrip train_model.py agar nilai R-squared RandomForestRegressor meningkat di atas 0.85. Pisahkan dataset sebelum melatih model regresi: gunakan hanya baris data normal (340 baris) untuk train_test_split dan pelatihan RandomForestRegressor dengan One-Hot Encoding pada kode_perlakuan. Tetap latih IsolationForest (contamination=0.15) pada seluruh 400 baris data berdasarkan fitur rasio penyusutan tiap tahap."
- Hasil Validasi Manusia:
  - Akurasi `RandomForestRegressor` meningkat signifikan menjadi `R² = 0.9800` dan `MAE = 0.3902%`.
  - `IsolationForest` mendeteksi tepat `60 / 400` batch anomali (15%).

  ## Sesi 3: Pembangunan REST API Backend (FastAPI)
- Tanggal: 23 November 2026
- Berkas Target: `main.py`
- Prompt Utama ke Kiro:
  > "Baca terlebih dahulu isi file train_model.py di folder ini agar nama kolom fitur dan prapemrosesan data sama persis, lalu buatkan kode Python lengkap pada file main.py menggunakan FastAPI untuk backend RUGI-BATCH AI dengan spesifikasi wajib: muat file model rf_rendemen.joblib dan iso_anomali.joblib, pasang middleware CORSMiddleware dengan allow_origins=['*'], buat endpoint GET /, POST /api/login dengan akun demo demo@rugibatch.id (sandi demo123), POST /api/analyze-batch untuk menghitung rendemen aktual, prediksi AI, status batch, tahap paling bocor, estimasi kerugian Rupiah, serta rekomendasi pemeriksaan, dan POST /api/sync-offline untuk memproses daftar batch tertunda saat luring."
- Audit & Intervensi Manual Manusia:
  - Menguji server lokal melalui Swagger UI (`http://127.0.0.1:8000/docs`) dan menemukan dua cacat logika bisnis pada kode awal Kiro: server tetap memunculkan peringatan kebocoran dan kerugian Rp 28.018,44 pada batch berstatus Normal, serta rumus kerugian batch Anomali hanya mengalikan selisih berat produk jadi dengan harga bahan baku mentah (Rp 508.078,44) tanpa memperhitungkan faktor konversi rendemen.

## Sesi 4: Kalibrasi Logika Bisnis dan Rumus Kerugian Finansial
- Tanggal: 23 November 2026
- Berkas Target: `main.py`
- Prompt Perbaikan ke Kiro:
  > "Perbaiki logika pada endpoint POST /api/analyze-batch di file main.py dengan dua aturan berikut. Pertama, jika status_batch bernilai Normal, atur nilai tahap_paling_bocor menjadi Tidak Ada, atur estimasi_kerugian_rupiah menjadi 0, dan isi rekomendasi_pemeriksaan dengan kalimat Produksi berjalan optimal sesuai standar rendemen. Pertahankan suhu dan durasi penggorengan saat ini. Kedua, jika status_batch bernilai Anomali, hitung selisih bobot produk jadi yang hilang, bagi angka tersebut dengan rasio prediksi rendemen normal untuk mendapatkan bobot bahan baku mentah yang terbuang, lalu kalikan dengan harga_per_kg untuk menghasilkan estimasi_kerugian_rupiah."
- Hasil Validasi Manusia:
  - Pengujian ulang skenario Normal menghasilkan status Normal dengan `tahap_paling_bocor` bernilai Tidak Ada dan `estimasi_kerugian_rupiah` sebesar `0`.
  - Pengujian ulang skenario Anomali pada Tahap 3 - Goreng menghasilkan `estimasi_kerugian_rupiah` sebesar `1544001.61` yang sesuai dengan perhitungan ekonomi produksi.

## Sesi 5: Penyelarasan Basis Data SQLite dan Endpoint Desain UI/UX Stitch
- Tanggal: 24 November 2026
- Berkas Target: `database.py` dan `main.py`
- Prompt Utama ke Kiro:
  > "Baca file database.py, file main.py, berkas DESIGN.md, dan semua file code.html di dalam folder stitch_rugi_batch_umkm_monitor. Perbarui file main.py agar terhubung ke database.py, simpan setiap hasil analisis ke rugi_batch.db, cocokkan seluruh nama field request dan response JSON dengan layar HTML tersebut, serta tambahkan endpoint GET /api/dashboard-summary dan GET /api/batches. Pertahankan pemuatan model rf_rendemen.joblib dan iso_anomali.joblib serta rumus kerugian yang sudah ada di main.py."
- Audit & Intervensi Manual Manusia:
  - Memeriksa kode hasil pembaruan Kiro pada `main.py` untuk memastikan vektor fitur Random Forest (9 kolom) dan Isolation Forest (4 rasio penyusutan) tetap identik dengan skrip pelatihan `train_model.py`.
  - Memverifikasi kesesuaian akun demo desain Stitch (`sari@bawanghangat.com`) dan akun demo kompetisi (`demo@rugibatch.id`) serta alias nama variabel output JSON agar cocok dengan antarmuka HTML.
  - Menghapus berkas cadangan sementara `main copy.py` serta mengeluarkan folder aset HTML `stitch_rugi_batch_umkm_monitor` dari direktori backend sebelum unggah ke repositori.

---

## Sumber Dataset, Pustaka, dan Aset Pihak Ketiga Backend
- Referensi Kalibrasi Dataset: Parameter rendemen normal bawang merah goreng (25,83% hingga 37,33%) dan variasi perlakuan mengacu pada data penelitian jurnal ilmiah Anggarkasih dkk. yang dibangkitkan menjadi 400 data simulasi terkalibrasi.
- Pustaka Pemrosesan Data dan AI: Python, Pandas, NumPy, Scikit-Learn (RandomForestRegressor dan IsolationForest), serta Joblib untuk serialisasi model.
- Pustaka API dan Basis Data: FastAPI, Uvicorn, Pydantic, dan SQLite3 (`rugi_batch.db`).
- Layanan Hosting Backend: PythonAnywhere (`https://abiliya.pythonanywhere.com`).
