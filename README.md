# Prediksi Flare Matahari dari Magnetogram SDO/HMI

Pipeline ini menjawab satu pertanyaan:

> **Melihat kondisi medan magnet sebuah wilayah aktif di Matahari hari ini,
> akankah wilayah itu melepaskan flare besar (kelas M atau lebih) dalam
> 24 jam ke depan?**

Jawabannya dibangun bertingkat — mulai dari tebakan paling bodoh, lalu naik
satu per satu ke model yang lebih pintar, sambil selalu mengukur apakah
kenaikan itu nyata. Setiap tingkat harus mengalahkan tingkat sebelumnya untuk
dianggap layak.

**Keluarannya:** tabel metrik (`output_dataset/metrics_test_*.csv`) dan delapan
grafik diagnostik (`output_dataset/figures/`).

---

## Daftar isi

1. [Istilah yang perlu diketahui dulu](#istilah-yang-perlu-diketahui-dulu)
2. [Mulai cepat](#mulai-cepat)
3. [Menjalankan ulang tanpa mengunduh](#menjalankan-ulang-tanpa-mengunduh)
4. [Mengekspor laporan PDF/HTML](#mengekspor-laporan-pdfhtml)
5. [Struktur proyek](#struktur-proyek)
6. [Cara membaca metrik](#cara-membaca-metrik)
7. [Hasil](#hasil)
8. [Kenapa dirancang seperti ini](#kenapa-dirancang-seperti-ini)
9. [Jebakan data yang sudah ditangani](#jebakan-data-yang-sudah-ditangani)
10. [Tingkat 4 — CNN magnetogram](#tingkat-4--cnn-magnetogram)
11. [Langkah berikutnya](#langkah-berikutnya)

---

## Istilah yang perlu diketahui dulu

| Istilah | Artinya |
|---|---|
| **Flare** | Ledakan di atmosfer Matahari. Kelasnya A → B → C → **M** → X; tiap naik satu huruf, kekuatannya **10× lipat**. Kelas M ke atas yang berbahaya bagi satelit dan jaringan listrik. |
| **AR** (*Active Region*) | Wilayah aktif — bercak medan magnet kuat di permukaan Matahari, tempat flare lahir. NOAA memberi nomor, mis. AR 11944. |
| **Magnetogram** | Peta medan magnet permukaan Matahari, difoto satelit SDO tiap 12 menit. |
| **HARP / SHARP** | Potongan magnetogram yang otomatis mengikuti satu wilayah aktif, **plus ~20 angka ringkasan fisisnya** (total fluks magnet, energi bebas, dsb). Angka-angka inilah fitur model kita. |
| **Basis rate** | Porsi sampel yang benar-benar berujung flare. Di sini hanya **~4%** — sangat timpang, dan ini mengubah cara kita menilai model. |
| **Horizon prakiraan** | Seberapa jauh ke depan kita memprediksi. Di sini 24 jam. |

Penjelasan lebih dalam soal data ada di [DATA_GUIDE.md](DATA_GUIDE.md).

---

## Mulai cepat

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# pertama kali: unduh data + bangun dataset + latih  (~10-20 menit)
.\venv\Scripts\python.exe run_pipeline.py --quick

# selanjutnya: hanya melatih model, tanpa jaringan  (~15 detik)
.\venv\Scripts\python.exe run_pipeline.py --quick --offline
```

Atau buka [flare_baseline.ipynb](flare_baseline.ipynb) untuk versi bernarasi
langkah demi langkah.

> **Satu-satunya langkah manual:** daftarkan email Anda ke JSOC.
> Gratis, sekali seumur hidup — lihat [DATA_GUIDE.md §1](DATA_GUIDE.md).

---

## Menjalankan ulang tanpa mengunduh

Data mentah **tidak pernah diunduh dua kali**. Cache-nya dua lapis:

| Lapis | Isinya | Lokasi |
|---|---|---|
| **Mentah** | keyword SHARP per bulan, katalog flare per tahun | `data/raw/*.parquet` |
| **Berlabel** | hasil quality control + windowing + pelabelan, siap latih | `data/processed/labeled_*.parquet` |

Pilih perintahnya sesuai yang ingin Anda lakukan:

| Yang ingin Anda lakukan | Perintah | Lama |
|---|---|---|
| Menguji model, data tidak berubah | `--quick --offline` | ~15 detik |
| Mengubah horizon / ambang kelas / rentang waktu | `--quick` (tanpa `--offline`) | ~1 menit |
| Mulai dari nol, `data/` kosong | `--quick` | ~10–20 menit |

Perintah tanpa `--offline` **tetap tidak mengunduh** selama `data/raw/` masih
ada — ia hanya membaca ulang dari disk lalu membangun label baru.

### Pengaman cache basi

Nama berkas cache berlabel memuat seluruh parameter yang mempengaruhi isinya:

```
labeled_20130101-20141231_cad4h_obs24h_lat0h_fcst24h_M_cmd70_vr3500_q1.parquet
        └── rentang ──┘  └cadence┘└obs┘└jeda┘└horizon┘└kelas┘└─── batas QC ───┘
```

Jadi kalau Anda mengubah salah satunya lalu menjalankan `--offline`, pipeline
**menolak jalan** alih-alih diam-diam memakai label dari konfigurasi lama:

```
[--offline] Cache yang cocok tidak ditemukan:
  data/processed/labeled_..._fcst48h_....parquet
Cache yang tersedia:
  labeled_..._fcst24h_....parquet
Jalankan sekali TANPA --offline untuk membangunnya.
```

---

## Mengekspor laporan PDF/HTML

```powershell
.\export_report.ps1          # PDF + HTML
.\export_report.ps1 -Html    # HTML saja, tanpa perlu LaTeX/pandoc
```

> **Jalankan notebook sampai selesai (Run All) lebih dulu.** Ekspor hanya
> menyalin output yang tersimpan di dalam `.ipynb`. Kalau notebook belum
> dijalankan, PDF-nya berisi kode tanpa satu pun hasil.

### Kalau ekspor PDF gagal

Pesan Jupyter *"you will need to install xelatex"* **menyesatkan**. Rantai
ekspor butuh dua program, dan yang biasanya hilang justru yang pertama:

| Program | Perannya | Cara memasang |
|---|---|---|
| **pandoc** | sel markdown → LaTeX | `winget install JohnMacFarlane.Pandoc` |
| **xelatex** (MiKTeX) | LaTeX → PDF | biasanya sudah ada |

Setelah memasang pandoc, **restart VS Code / Jupyter** — proses yang sudah
berjalan tidak melihat perubahan PATH. Lewat `export_report.ps1` tidak perlu
restart, karena skrip itu memuat ulang PATH sendiri.

Jalur cadangan yang selalu berhasil: buka `flare_baseline.html` di browser,
lalu Ctrl+P → *Save as PDF*.

> Log LaTeX memunculkan `No counter 'none' defined` beberapa kali. Itu
> ketidakcocokan kosmetik antara template nbconvert dan tcolorbox versi baru.
> Isi PDF-nya tetap lengkap dan benar.

---

## Struktur proyek

```
flare_pipeline/
  config.py        Konfigurasi + daftar keyword SHARP
  acquisition.py   Unduhan JSOC (drms) + HEK (sunpy), di-cache per bulan
  labeling.py      Konversi kelas GOES <-> fluks, indeks flare per AR
  dataset.py       Quality control, windowing, fitur, split kronologis
  models.py        Tangga model 0 -> 3 + kalibrasi probabilitas
  evaluate.py      Matriks evaluasi + grafik diagnostik
  magnetogram.py   Unduhan & pembacaan citra .fits
  cnn.py           Tingkat 4: cache piksel + CNN magnetogram (butuh PyTorch)

run_pipeline.py         Driver CLI
flare_baseline.ipynb    Driver notebook (bernarasi)
export_report.ps1       Ekspor notebook -> PDF / HTML
DATA_GUIDE.md           Panduan data, alternatif open-source, troubleshooting

data/            Cache unduhan (mentah + berlabel)
output_dataset/  Metrik + grafik hasil
magnetogram_fits/  Citra .fits, baru dipakai di tingkat CNN
```

---

## Cara membaca metrik

Ini bagian terpenting, karena **akurasi menyesatkan total di sini.**

Dengan basis rate 4%, model yang selalu menjawab *"tidak akan ada flare"*
mencapai **akurasi 96%** — terdengar hebat, padahal tidak berguna sama sekali.
Karena itu domain space weather memakai metrik lain:

| Singkatan | Nama lengkap | Artinya dalam satu kalimat | Nilai kalau asal tebak |
|---|---|---|---|
| **TSS** | True Skill Statistic | **Metrik utama.** Seberapa jauh model unggul dari tebakan acak, tanpa terpengaruh betapa langkanya flare. | 0 |
| **HSS** | Heidke Skill Score | Mirip TSS, tapi **menghukum kelebihan peringatan**. Lebih relevan untuk pemakaian nyata. | 0 |
| **POD** | Probability of Detection | Dari semua flare yang benar-benar terjadi, berapa persen berhasil tertangkap. | — |
| **FAR** | False Alarm Ratio | Dari semua peringatan yang dikeluarkan, berapa persen ternyata palsu. | — |
| **PR-AUC** | Precision-Recall AUC | Mutu peringkatan model pada data timpang. Lebih jujur daripada ROC-AUC di sini. | = basis rate (0,047) |
| **BSS** | Brier Skill Score | Seberapa baik probabilitasnya dibanding sekadar menebak basis rate. **Negatif = kalah dari tebakan konstan.** | 0 |

Singkatnya: **TSS dan HSS makin tinggi makin baik (maksimum 1,0); accuracy
abaikan saja.**

---

## Hasil

Preset `--quick`: latih 2013, uji paruh kedua 2014. Set uji berisi 4.510 sampel
dengan basis rate 4,66%. Ambang keputusan dipilih di data validasi, lalu
dibekukan sebelum menyentuh data uji.

| Model | TSS ↑ | HSS ↑ | POD | FAR ↓ | PR-AUC ↑ | BSS ↑ | Accuracy |
|---|---|---|---|---|---|---|---|
| 3a. Random Forest | **0,684** | 0,257 | 0,862 | 0,809 | 0,300 | 0,194 | 0,824 |
| 2. Regresi logistik | 0,683 | 0,219 | 0,905 | 0,834 | 0,572 | 0,322 | 0,784 |
| 3b. Gradient Boosting | 0,668 | 0,274 | 0,824 | 0,795 | 0,242 | 0,128 | 0,843 |
| 1. Ambang satu fitur | 0,663 | 0,192 | 0,919 | 0,851 | 0,530 | 0,344 | 0,752 |
| 0b. Persistence | 0,550 | **0,545** | 0,571 | 0,439 | 0,340 | 0,088 | 0,959 |
| 0a. Climatology | 0,000 | 0,000 | 0,000 | — | 0,047 | 0,000 | **0,953** |

### Tiga hal yang perlu Anda tangkap dari tabel ini

**1. Accuracy tertinggi justru milik model terburuk.**
Climatology tidak pernah memprediksi flare sama sekali, tapi accuracy-nya 0,953 —
kedua tertinggi. Ini bukti langsung kenapa accuracy tidak boleh dipakai.

**2. Persistence menang telak di HSS meski TSS-nya rendah.**
Persistence hanya beraturan *"AR yang baru saja flare akan flare lagi"*.
Model ML mengalahkannya di TSS (0,68 vs 0,55) tapi kalah jauh di HSS
(0,26 vs 0,55). Sebabnya: ambang yang dioptimalkan untuk TSS membuat model
memberi peringatan **4× lebih sering** daripada seharusnya. TSS tidak
menghukum kelebihan peringatan; HSS menghukumnya.

> **Implikasi praktis:** kalau target Anda pemakaian operasional (jumlah
> peringatan wajar), optimalkan ambang terhadap HSS, bukan TSS —
> `ev.best_threshold(y_val, p_val, "HSS")`.

**3. Model rumit nyaris tidak mengungguli regresi logistik.**
Random Forest 0,684 vs regresi logistik 0,683 — selisihnya jauh di dalam
selang kepercayaan. Ini konsisten dengan literatur, bukan tanda ada yang salah:
sinyal dominan pada parameter SHARP memang mendekati linear dalam ruang log.

Semua angka di atas dari 2 tahun data. Rentang penuh (`Config()`, 2010–2018)
memberi ~10× lebih banyak kejadian positif dan selang kepercayaan yang jauh
lebih sempit.

### Efek menambahkan fitur evolusi temporal

`--temporal --history` menambahkan laju perubahan dan variabilitas 24 jam per
HARP, plus riwayat flare AR itu sendiri. Jumlah fitur naik dari 18 menjadi 55.

| Model | TSS dasar | TSS + temporal | Perubahan |
|---|---|---|---|
| Random Forest | 0,684 | **0,740** | naik |
| Gradient Boosting | 0,668 | 0,690 | naik sedikit |
| Regresi logistik | 0,683 | 0,607 | **turun** |

Model pohon memanfaatkan fitur tambahan; regresi logistik justru memburuk.
Penyebabnya 55 fitur yang saling berkorelasi kuat dengan hanya 11 ribu baris
latih — melampaui kapasitas model linear tanpa penyetelan regularisasi. Kalau
Anda menempuh jalur ini, setel `C` pada `make_logistic()` atau ganti ke
penalti L1.

Setiap kombinasi fitur menulis ke berkas dan subfolder grafiknya sendiri
(`metrics_test_base.csv` vs `metrics_test_temporal_history.csv`), jadi hasilnya
tidak saling menimpa.

---

## Kenapa dirancang seperti ini

### Split kronologis, bukan acak

Ini keputusan paling penting di seluruh proyek.

Sampel dari AR yang sama pada jam-jam berdekatan nyaris identik. `train_test_split`
acak akan menaruh tetangga langsung dari setiap baris uji ke dalam set latih —
model "sudah pernah melihat" jawabannya. Skornya melonjak dan **model tampak
jauh lebih baik daripada kenyataannya.**

Karena itu pembagiannya murni menurut waktu: latih pada masa lalu, uji pada masa
depan, persis seperti pemakaian operasionalnya.

> TSS di atas ~0,9 pada tugas ini hampir selalu gejala kebocoran semacam ini.

### Persistence sebagai lawan yang sesungguhnya

Baseline yang pantas dikalahkan bukan tebakan acak, melainkan aturan sederhana
*"AR yang baru saja flare akan flare lagi"*. Model ML yang tidak mengalahkannya
tidak menambah nilai apa pun — betapa pun canggih arsitekturnya.

### Kalibrasi dipisahkan dari pemilihan ambang

`class_weight="balanced"` sengaja mendistorsi probabilitas keluaran agar ambang
bergeser. Akibatnya Brier dan BSS jadi tak bermakna — BSS sempat **−1,7**.

Kalibrasi isotonik memulihkan makna probabilistiknya tanpa mengubah urutan skor,
jadi TSS dan AUC tetap sama persis. Kalibrasi dilakukan di **paruh pertama**
validasi, pemilihan ambang di **paruh kedua** — memakai potongan yang sama untuk
keduanya akan membuat ambangnya tampak lebih baik daripada kenyataan.

### Bootstrap blok per HARP

Selang kepercayaan dihitung dengan mengambil ulang sampel **per wilayah aktif**,
bukan per baris. Baris-baris dari AR yang sama sangat berkorelasi; bootstrap per
baris menghasilkan selang yang terlalu sempit dan membuat kita terlalu percaya
diri pada hasilnya.

### Keyword SHARP, bukan piksel mentah

Keyword SHARP *adalah* ringkasan fisis magnetogram yang sudah dihitung JSOC dari
piksel yang sama. Mengunduh citranya (~1 TB/tahun) baru perlu di tingkat CNN.

---

## Jebakan data yang sudah ditangani

Empat masalah ini punya sifat yang sama dan berbahaya: **semuanya gagal secara
senyap.** Tidak ada exception, tidak ada pesan error — hanya hasil yang kosong
atau salah, yang baru ketahuan jauh di hilir.

| # | Jebakan | Akibatnya |
|---|---|---|
| 1 | JSOC menolak format waktu ISO-8601 | Mengembalikan **tabel kosong**, bukan error. Query harus `2014.01.01_00:00:00_TAI`. |
| 2 | `LON_FWT` disangka bujur Carrington | Sebenarnya bujur **Stonyhurst**, yang sudah setara CMD. Menguranginya dengan `CRLN_OBS` membuang hampir semua data. |
| 3 | Cadence 6 atau 12 jam | Selalu jatuh di 06:00 & 18:00 TAI — jam manuver kalibrasi SDO. Kehilangan **separuh data**. Pakai 1h, 2h, atau 4h. |
| 4 | `TOTBSQ` tidak ada di seri CEA | JSOC mengembalikannya sebagai kolom NaN penuh, lalu `dropna` menghapus **seluruh dataset**. |

Gejala dan solusi lengkapnya ada di [DATA_GUIDE.md §5](DATA_GUIDE.md).

---

## Tingkat 4 — CNN magnetogram

Tingkat 0–3 memakai 18 keyword SHARP, yang *sudah* merupakan ringkasan fisis
buatan manusia atas magnetogram. Tingkat 4 menanyakan satu hal spesifik:
**adakah informasi di dalam piksel yang tidak tertangkap oleh 18 angka itu?**
Kandidat utamanya adalah bentuk dan ketajaman garis pembalikan polaritas (PIL)
serta tata letak spasial fluks — keduanya tidak punya padanan skalar tunggal.

Kodenya di [flare_pipeline/cnn.py](flare_pipeline/cnn.py).

### Prasyarat

PyTorch **tidak** termasuk `requirements.txt`, karena tingkat 0–3 tidak
membutuhkannya sama sekali:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Menjalankan

```bash
# 1) unduh citra sebagai blok AR kontigu (bertahap; ulangi untuk melanjutkan)
python run_pipeline.py --offline --cnn --cnn-ar-blocks 40 --cnn-download 2000

# 2) latih seluruh tangga 0-4 pada citra yang sudah ada di disk
python run_pipeline.py --offline --cnn --cnn-ar-blocks 40

# 3) varian: keyword SHARP digabung ke kepala klasifikasi CNN
python run_pipeline.py --offline --cnn --cnn-ar-blocks 40 --cnn-scalars
```

### Yang sebenarnya membatasi unduhan: jumlah permintaan, bukan jumlah berkas

Ini bagian yang paling mahal dipelajari sendiri, jadi ada baiknya dibaca sebelum
menjalankan unduhan besar.

**JSOC hanya mengizinkan satu permintaan ekspor tertunda per akun**, dan setiap
permintaan butuh beberapa menit untuk diproses. Konsekuensinya: 800 berkas dalam
800 permintaan memakan puluhan jam, sementara 800 berkas dalam 40 permintaan
memakan sekitar satu jam. Ukuran unduhannya persis sama.

Karena itu ada dua cara memilih citra:

| | `--cnn-ar-blocks N` | (default, frame tersebar) |
|---|---|---|
| Bentuk sampel | N wilayah aktif utuh, frame berurutan | frame tunggal dari banyak AR |
| Permintaan JSOC | ~N | ratusan |
| Keragaman AR | rendah | tinggi |
| Cocok untuk | unduhan nyata, dan prasyarat ConvLSTM | dataset yang citranya sudah lengkap di disk |

Permintaan digabungkan memakai **spesifikasi rentang waktu**
(`[harp][t0-t1@4h]`), bukan daftar record yang dipisah koma. Selain jauh lebih
pendek, bentuk itulah yang memang dirancang JSOC untuk ekspor massal.

Dua jebakan lain yang sudah ditangani `download_patches()`, keduanya gagal
dengan cara yang membingungkan:

- Batch yang terlalu panjang ditolak sebagai `HTTP 414` (batas URL) atau
  `Record-set specification is too long [status=4]` (batas JSOC, jauh lebih
  ketat). Ukuran batch karena itu dihitung dari panjang spec, bukan jumlah record.
- Permintaan yang **gagal** pun bisa tetap terhitung "tertunda" dan memblokir
  seluruh akun sampai JSOC melepaskannya — termasuk memblokir `url_quick`,
  sehingga tidak ada jalan pintas. Yang bisa dilakukan hanya menunggu dan
  mencoba lagi, dan itu yang dilakukan pipeline ini.

### Tiga keputusan desain yang menentukan kesahihan hasilnya

**Tidak semua baris diunduh citranya.** Mengunduh seluruh dataset penuh berarti
puluhan sampai ratusan GB. `select_patch_subset()` mengambil semua sampel
positif plus negatif secukupnya, **distratifikasi per bulan**. Stratifikasi itu
bukan hiasan: negatif yang diambil acak murni akan menumpuk di periode maksimum
siklus, dan CNN bisa belajar membedakan "tahun ramai" dari "tahun sepi"
alih-alih membedakan AR yang akan flare.

**Seluruh tangga model dibatasi ke baris yang sama.** Kalau CNN dinilai pada
3.000 baris test sementara Random Forest dinilai pada 9.000 baris test dengan
komposisi berbeda, kedua angka TSS itu tidak bisa disandingkan sama sekali.
`restrict_to_cache()` memangkas train/val/test untuk semua tingkat sekaligus.
`--cnn-no-restrict` mematikannya, dan pipeline akan mencetak peringatan bahwa
angkanya tidak lagi sebanding.

**Normalisasi arcsinh, bukan pembagian linear.** Bz membentang dari ~10 G di
medan tenang sampai ~3000 G di umbra. `arcsinh(Bz/50)` bersifat linear di
bawah 50 G dan logaritmik di atasnya, sehingga struktur lemah di sekitar PIL
tidak tertelan umbra — dan **tandanya tetap terjaga**, yang wajib, karena
polaritas adalah inti fisika PIL.

### Kenapa hasilnya kemungkinan besar mengecewakan

CNN ini punya ~10⁵ parameter. Preset `--quick` hanya menyediakan ratusan sampel
positif. Modul ini sudah melawan overfitting dengan arsitektur kecil,
augmentasi cermin, `pos_weight` pada loss, dan early stopping pada potongan
**kronologis** terakhir set latih — tapi tidak ada teknik yang bisa menciptakan
informasi yang tidak ada di data.

Periksa `output_dataset/figures/<tag>/cnn_training_curve.png`: kalau loss latih
terus turun sementara skor pantau mandek, yang Anda lihat adalah hafalan, bukan
pembelajaran. Kalau CNN kalah dari Random Forest, jawaban yang benar hampir
selalu "datanya kurang", bukan "arsitekturnya kurang dalam".

---

## Langkah berikutnya

Berurutan dari yang paling murah dan paling besar dampaknya:

1. **Perpanjang rentang data** — jalankan `run_pipeline.py` tanpa `--quick`
   (2010–2018). Ini yang paling menyempitkan selang kepercayaan.
2. **Tambahkan fitur evolusi** — `--temporal --history`.
3. **Bandingkan dengan SWAN-SF** ([DATA_GUIDE.md §3a](DATA_GUIDE.md)) supaya
   angka Anda bisa disandingkan langsung dengan literatur.
4. **Baru kemudian CNN** ([tingkat 4](#tingkat-4--cnn-magnetogram)) di atas
   citra magnetogram.

> Naik ke CNN hanya masuk akal kalau tingkat 3 sudah mengalahkan tingkat 2
> secara meyakinkan. Kalau gradient boosting saja belum mengungguli regresi
> logistik, CNN dengan data sebanyak ini hampir pasti akan overfit.
#   F l a r e - P r e d i c t i o n  
 