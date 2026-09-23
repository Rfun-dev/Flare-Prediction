# Panduan Data

Dokumen ini menjawab tiga hal: **data apa yang dipakai, dari mana asalnya, dan
apa yang harus Anda kerjakan sendiri.**

Kabar baiknya: hampir semuanya otomatis. Hanya **satu** langkah yang benar-benar
butuh tindakan manual Anda — mendaftarkan email ke JSOC (§1).

---

## Daftar isi

- [§0 — Kenapa "magnetogram" di sini berarti SHARP](#0--kenapa-magnetogram-di-sini-berarti-sharp)
- [§1 — Langkah manual: daftarkan email ke JSOC](#1--langkah-manual-daftarkan-email-ke-jsoc)
- [§2 — Langkah otomatis: unduhan dijalankan pipeline](#2--langkah-otomatis-unduhan-dijalankan-pipeline)
- [§3 — Alternatif open-source](#3--alternatif-open-source)
- [§4 — Mengunduh citra magnetogram (tahap CNN)](#4--mengunduh-citra-magnetogram-tahap-cnn)
- [§5 — Masalah yang sering muncul](#5--masalah-yang-sering-muncul)

### Singkatan yang dipakai di dokumen ini

| Singkatan | Kepanjangan | Artinya |
|---|---|---|
| **SDO** | Solar Dynamics Observatory | Satelit NASA pengamat Matahari. |
| **HMI** | Helioseismic and Magnetic Imager | Instrumen di SDO yang memotret medan magnet. |
| **JSOC** | Joint Science Operations Center | Server Stanford tempat data HMI disimpan. |
| **AR** | Active Region | Wilayah aktif — bercak medan magnet kuat, tempat flare lahir. |
| **HARP** | HMI Active Region Patch | Kotak yang otomatis mengikuti satu AR sepanjang hidupnya. |
| **SHARP** | Space-weather HARP | HARP + ~20 parameter fisis siap pakai. **Ini yang kita gunakan.** |
| **HEK** | Heliophysics Event Knowledgebase | Basis data kejadian Matahari, termasuk katalog flare. |
| **CMD** | Central Meridian Distance | Jarak sudut AR dari garis tengah cakram Matahari. |
| **TAI** | International Atomic Time | Skala waktu yang dipakai JSOC (bukan UTC). |
| **Cadence** | — | Jarak waktu antar sampel, mis. "cadence 4 jam". |

---

## §0 — Kenapa "magnetogram" di sini berarti SHARP

Magnetogram mentah HMI berukuran 4096×4096 piksel, diambil tiap 12 menit —
sekitar **1 TB per tahun**. Melatih model klasik langsung di atas piksel sebesar
itu boros dan tidak perlu.

Sebabnya: tim SDO sudah menerbitkan **SHARP**, yaitu potongan magnetogram per
wilayah aktif **beserta ~20 parameter fisis yang mereka hitung langsung dari
piksel magnetogram tersebut** — total fluks magnet tak bertanda, helisitas arus,
energi bebas medan, panjang garis pembalikan polaritas, dan seterusnya.

Jadi ketika pipeline ini memakai "keyword SHARP", ia **sedang memakai data
magnetogram** — hanya sudah diringkas oleh pihak yang paling berwenang
meringkasnya.

| Bentuk data | Isinya | Ukuran | Dipakai di tingkat |
|---|---|---|---|
| **Keyword SHARP** (angka) | ~20 ringkasan fisis per AR per waktu | ~50 MB / 9 tahun | **0–3** (semua model di pipeline ini) |
| **Cutout SHARP** (`.fits`) | citra magnetogram per AR | ~0,6 MB / berkas | 4 (CNN / ConvLSTM) |

> **Kesimpulan praktis:** dari baseline sampai gradient boosting, Anda **tidak
> perlu mengunduh satu pun citra.** Citra baru relevan saat naik ke CNN.

### Dua sumber data yang digabungkan

```
  JSOC (Stanford)                      HEK / GOES
  hmi.sharp_cea_720s                   katalog kejadian flare
  = FITUR (X)                          = LABEL (y)
        |                                    |
        |  HARPNUM --> NOAA_AR ------------- |
        +---------------+--------------------+
                        v
              satu baris per (AR, waktu)
```

Kuncinya ada di panah tengah: setiap kejadian flare di HEK menyertakan nomor
NOAA AR, dan setiap HARP tahu nomor NOAA AR-nya. Itulah yang memungkinkan fitur
dan label dijodohkan.

---

## §1 — Langkah manual: daftarkan email ke JSOC

Ini satu-satunya langkah yang tidak bisa diotomatiskan. **Gratis, sekali seumur
hidup, sekitar 5 menit.**

**Langkah 1.** Buka <http://jsoc.stanford.edu/ajax/register_email.html>

**Langkah 2.** Isi alamat email Anda, klik **Submit**.

**Langkah 3.** Buka inbox, klik tautan konfirmasi dari JSOC.

**Langkah 4.** Masukkan email tersebut ke
[flare_pipeline/config.py](flare_pipeline/config.py):

```python
jsoc_email: str = "email.anda@contoh.com"
```

Atau lewat argumen baris perintah:

```powershell
.\venv\Scripts\python.exe run_pipeline.py --email email.anda@contoh.com
```

**Langkah 5.** Verifikasi berhasil:

```powershell
.\venv\Scripts\python.exe -c "import drms; print(drms.Client(email='email.anda@contoh.com').series(r'hmi\.sharp'))"
```

Kalau muncul daftar seri (`hmi.sharp_720s`, `hmi.sharp_cea_720s`, ...), Anda
sudah siap.

> **Catatan:** email sebenarnya hanya wajib untuk **ekspor citra**
> (`client.export`). Query keyword (`client.query`) jalan tanpa registrasi.
> Tapi daftarkan saja sekarang, supaya tahap CNN nanti tidak terhambat.

---

## §2 — Langkah otomatis: unduhan dijalankan pipeline

Setelah email terdaftar, seluruh unduhan ditangani sendiri oleh pipeline:

```powershell
# uji cepat: 2 tahun, cadence 4 jam   (~10-20 menit, sekali saja)
.\venv\Scripts\python.exe run_pipeline.py --quick

# produksi: 2010-2018, cadence 1 jam  (~1-3 jam, tergantung antrean JSOC)
.\venv\Scripts\python.exe run_pipeline.py

# hanya melatih model, tanpa jaringan (~15 detik)
.\venv\Scripts\python.exe run_pipeline.py --quick --offline
```

**Proses boleh Anda hentikan kapan saja dengan Ctrl-C.** Unduhan di-cache per
bulan, jadi menjalankannya lagi akan melanjutkan dari bulan terakhir yang
selesai — bukan mengulang dari awal.

### Cache berlapis dua

| Lapis | Isinya | Lokasi | Dibangun ulang kalau... |
|---|---|---|---|
| **Mentah** | keyword SHARP per bulan, katalog flare per tahun | `data/raw/*.parquet` | berkasnya hilang |
| **Berlabel** | hasil quality control + windowing + pelabelan | `data/processed/labeled_*.parquet` | parameter pelabelan berubah |

Nama berkas lapis kedua memuat sidik jari konfigurasinya:

```
labeled_20130101-20141231_cad4h_obs24h_lat0h_fcst24h_M_cmd70_vr3500_q1.parquet
        └── rentang ──┘  └cadence┘└obs┘└jeda┘└horizon┘└kelas┘└─── batas QC ───┘
```

Karena itu `--offline` **menolak jalan** — bukan diam-diam memakai label lama —
kalau Anda mengubah horizon prakiraan, ambang kelas, cadence, rentang waktu,
atau batas quality control.

### Perkiraan volume

| Rentang | Cadence | Baris SHARP | Ukuran cache |
|---|---|---|---|
| 2 tahun (`--quick`) | 4 jam | ~120.000 | ~25 MB |
| 2010–2018 (penuh) | 1 jam | ~2.000.000 | ~400 MB |

---

## §3 — Alternatif open-source

Berguna dalam dua situasi: JSOC sedang bermasalah, atau Anda ingin pembanding
yang diakui literatur.

### §3a — SWAN-SF (paling disarankan sebagai pembanding)

**Space Weather ANalytics for Solar Flares.** Dataset ini sudah jadi sepenuhnya:
24 parameter SHARP dalam bentuk deret waktu multivariat, sudah diberi label
flare, sudah dibagi menjadi 5 partisi yang seimbang tingkat aktivitasnya.

Inilah dataset yang dipakai untuk membandingkan model antar-makalah ilmiah.

- **Unduh:** <https://doi.org/10.7910/DVN/EBCFKM> — Harvard Dataverse, ~2 GB,
  gratis, tanpa akun
- **Makalah:** Angryk et al. (2020), *Scientific Data* 7, 227
- **Cara pakai:** ekstrak ke `data/swan_sf/`. Tiap partisi berisi CSV per AR
  dengan nama kolom yang **sama persis** dengan `SHARP_FEATURES` di
  [flare_pipeline/config.py](flare_pipeline/config.py) — jadi `models.py` dan
  `evaluate.py` bisa langsung dipakai tanpa perubahan apa pun.

| | |
|---|---|
| **Kelebihan** | Angka Anda bisa dibandingkan langsung dengan literatur. Tidak butuh JSOC sama sekali. |
| **Kekurangan** | Partisinya dibagi menurut *span waktu* dan sudah baku, jadi Anda tidak bisa mengatur horizon prakiraan sendiri. |

### §3b — Katalog flare (alternatif untuk label)

Pipeline memakai **HEK**, karena setiap kejadian di sana sudah menyertakan
`ar_noaanum` — kunci untuk menjodohkan flare ke HARP.

| Sumber | Cara akses | Catatan |
|---|---|---|
| **HEK** (dipakai) | otomatis via `sunpy.net.Fido` | ada asosiasi NOAA AR |
| NOAA SWPC event list | [ngdc.noaa.gov](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/) | teks mentah, perlu parser sendiri |
| GOES XRS 1-menit | otomatis via `a.Instrument.xrs` | fluks kontinu, **tanpa** asosiasi AR |

### §3c — Citra siap-CNN

| Dataset | Tautan | Catatan |
|---|---|---|
| **SDO Benchmark** | <https://github.com/i4Ds/SDOBenchmark> | cutout AR multi-kanal, sudah dibagi train/test — **paling praktis untuk memulai CNN** |
| **SDOML** | <https://sdoml.org> | seluruh cakram Matahari, ter-normalisasi, ~6,5 TB |
| Cutout SHARP | pipeline ini, `magnetogram.download_patches()` | Anda kendalikan sendiri rentang & cadence-nya |

---

## §4 — Mengunduh citra magnetogram (tahap CNN)

Hanya perlu kalau Anda sudah siap naik ke tingkat 4.

### Cara termudah: lewat CLI

```bash
# unduh bertahap; ulangi perintahnya untuk melanjutkan dari tempat berhenti
python run_pipeline.py --cnn --cnn-ar-blocks 40 --cnn-download 2000
```

> **Pakai `--cnn-ar-blocks`, jangan frame tersebar.** JSOC hanya mengizinkan
> **satu permintaan ekspor tertunda per akun**, dan tiap permintaan butuh
> beberapa menit. Yang menentukan lama unduhan adalah jumlah permintaan, bukan
> jumlah berkas. 800 frame tersebar di 330 AR butuh ratusan permintaan
> (puluhan jam); 800 frame dari 40 AR kontigu butuh sekitar 40 (~1 jam).
> Ukuran unduhannya sama persis.
>
> Bonus ilmiahnya: frame berurutan memperlihatkan evolusi AR, yang tidak bisa
> dilihat dari snapshot tunggal — dan itu prasyarat kalau nanti naik ke ConvLSTM.

### Cara manual, kalau ingin mengendalikan subsetnya sendiri

```python
import pandas as pd
from flare_pipeline.config import quick_config
from flare_pipeline.magnetogram import download_patches, dedupe_download_dir
from flare_pipeline.cnn import build_patch_cache, missing_patches, select_patch_subset

cfg = quick_config()
df = pd.read_parquet(cfg.labeled_dataset_path())

subset = select_patch_subset(df, cfg, neg_per_pos=3)   # semua positif + negatif terstratifikasi
download_patches(missing_patches(subset, cfg).head(2000), cfg)

# FITS -> array float16 seragam, sekali saja; latihan berikutnya membaca ini
cache = build_patch_cache(subset, cfg, size=(128, 256))
```

> **Kenapa cache piksel dan bukan baca FITS langsung tiap epoch:** dekode FITS
> jauh lebih lambat daripada membaca memmap. Tanpa cache, waktu latih akan
> didominasi I/O, bukan komputasi. Cache-nya bisa ditambah tanpa dibangun ulang.

> **Kenapa `drms` bisa menggandakan isi folder:** ia tidak menimpa berkas yang
> sudah ada, melainkan menambah akhiran `.1`, `.2`, `.3`. Menjalankan skrip
> unduhan dua kali akan melipatgandakan isi folder tanpa peringatan apa pun.
>
> `download_patches()` di pipeline ini sudah memeriksa keberadaan berkas lebih
> dulu sehingga tidak mengulangi masalah itu. Untuk membersihkan duplikat yang
> terlanjur ada, panggil `dedupe_download_dir()`.

---

## §5 — Masalah yang sering muncul

### Gejala paling berbahaya: gagal tanpa pesan error

Empat masalah pertama di bawah punya sifat yang sama — **tidak ada exception,
tidak ada pesan error.** Pipeline berjalan sampai selesai dan menghasilkan
angka, tapi angkanya salah atau kosong.

---

**Gejala:** `[SHARP] ... 0 baris`, tapi tidak ada error apa pun.

- **Sebab:** format waktu ISO-8601 dikirim ke JSOC. JSOC tidak melempar error —
  ia hanya mengembalikan tabel kosong.
- **Solusi:** waktu harus berformat `2014.01.01_00:00:00_TAI`. Sudah ditangani
  otomatis oleh `acquisition.to_jsoc_time()`.

---

**Gejala:** setengah data hilang setelah filter QUALITY.

- **Sebab:** cadence 6 atau 12 jam selalu jatuh tepat di 06:00 dan 18:00 TAI —
  jam manuver kalibrasi harian SDO, ketika `QUALITY != 0` pada 100% baris.
- **Solusi:** pakai cadence 1, 2, atau 4 jam.

---

**Gejala:** semua AR terbuang oleh filter CMD.

- **Sebab:** `LON_FWT` disangka bujur Carrington lalu dikurangi `CRLN_OBS`.
- **Solusi:** `LON_FWT` adalah bujur **Stonyhurst**, yang menurut definisinya
  sudah diukur dari meridian tengah — jadi ia **sudah** CMD. Pakai apa adanya.

---

**Gejala:** seluruh dataset kosong setelah `dropna`.

- **Sebab:** ada keyword yang tidak tersedia di seri yang dipakai (`TOTBSQ` di
  seri CEA, misalnya). JSOC mengembalikannya sebagai kolom NaN penuh.
- **Solusi:** sudah ada penjaga di `quality_control()` yang mendeteksi kolom
  NaN 100% dan memberi peringatan alih-alih menghapus semuanya.

---

### Gejala lain

**Gejala:** `KeyError: 'true_bin'` atau kolom hilang.

- **Sebab:** gejala turunan dari data kosong di hulu — bukan masalah sebenarnya.
- **Solusi:** pipeline sekarang gagal cepat dengan pesan jelas di tahap
  akuisisi, sebelum sampai ke tahap ini.

---

**Gejala:** query JSOC timeout.

- **Sebab:** rentang waktu terlalu panjang untuk satu permintaan.
- **Solusi:** sudah dipotong per bulan, dengan 3× percobaan ulang otomatis.

---

**Gejala:** ~19% baris terbuang karena NaN.

- **Sebab:** `MEANPOT`, `SHRGT45`, dan sejenisnya butuh inversi medan vektor
  yang tidak selalu tersedia, terutama untuk AR lemah.
- **Solusi:** ini **normal**, bukan bug. Jumlahnya terlihat di jejak QC.

---

**Gejala:** banyak flare dengan `ar_noaanum == 0` di HEK.

- **Sebab:** flare kecil sering tidak diasosiasikan ke AR mana pun.
- **Solusi:** kejadian ini dibuang dari pelabelan per-AR, dan jumlahnya tetap
  dilaporkan supaya Anda tahu berapa banyak yang terlewat.
