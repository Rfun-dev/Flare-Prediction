"""Tingkat 4 — CNN langsung di atas piksel magnetogram.

Tingkat 0-3 bekerja pada keyword SHARP, yang *sudah* merupakan ringkasan fisis
buatan manusia dari magnetogram. Pertanyaan tingkat 4: adakah informasi di dalam
citra yang TIDAK tertangkap oleh 18 angka ringkasan itu — misalnya bentuk dan
ketajaman garis pembalikan polaritas (PIL), atau tata letak spasial fluks yang
tidak punya padanan skalar.

Alurnya tiga langkah, sengaja dipisah karena biayanya sangat berbeda:

  1. `select_patch_subset`  pilih baris mana yang citranya perlu diunduh.
                            Mengunduh semuanya = belasan GB, dan hampir selalu
                            mubazir karena ~96% baris berlabel negatif.
  2. `build_patch_cache`    FITS -> array float16 seragam dalam satu berkas
                            biner. Dekode FITS jauh lebih lambat daripada
                            memmap; tanpa cache, tiap epoch akan didominasi I/O.
  3. `CNNClassifier`        latih & prediksi, dengan antarmuka `fit` /
                            `predict_proba` yang sama seperti tingkat 0-3
                            sehingga `evaluate.py` memperlakukannya identik.

PERINGATAN YANG TIDAK BOLEH DIABAIKAN
-------------------------------------
CNN ini punya ~10^5 parameter, sementara set latih preset `--quick` hanya
memuat ratusan sampel positif. Rasio itu adalah resep overfitting. Modul ini
melawannya dengan arsitektur kecil, augmentasi, bobot kelas, dan early
stopping — tapi tidak ada teknik yang bisa menciptakan informasi yang tidak ada
di data. Kalau CNN kalah dari Random Forest, jawaban yang benar hampir selalu
"datanya kurang", bukan "arsitekturnya kurang dalam".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .magnetogram import patch_filename, read_patch, resize_patch

# Skala kompresi arcsinh untuk Bz (Gauss). Medan tenang ~10 G, umbra ~3000 G.
# arcsinh(x/B_KNEE) linear di bawah B_KNEE dan logaritmik di atasnya: struktur
# lemah di sekitar PIL tidak tertelan oleh umbra, dan tandanya (polaritas) tetap.
B_KNEE = 50.0
B_CLIP = 2500.0
_B_NORM = float(np.arcsinh(B_CLIP / B_KNEE))

DEFAULT_SIZE = (128, 256)   # (tinggi, lebar) — rasio khas bounding box HARP CEA


def normalize_bz(arr: np.ndarray) -> np.ndarray:
    """Bz Gauss -> kira-kira [-1, 1], monoton dan mempertahankan tanda."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0,
                      posinf=B_CLIP, neginf=-B_CLIP)
    a = np.clip(a, -B_CLIP, B_CLIP)
    return (np.arcsinh(a / B_KNEE) / _B_NORM).astype(np.float32)


# ==============================================================================
# 1. Memilih baris yang citranya layak diunduh
# ==============================================================================

def select_patch_subset(df: pd.DataFrame, cfg: Config,
                        neg_per_pos: int = 3, max_rows: int | None = None,
                        random_state: int | None = None) -> pd.DataFrame:
    """Ambil SEMUA positif + sampel negatif yang tersebar merata sepanjang waktu.

    Kenapa tidak semua baris: pada preset penuh ada ratusan ribu baris x 0,6 MB,
    yaitu puluhan sampai ratusan GB.

    Kenapa negatifnya distratifikasi per bulan dan bukan diambil acak begitu
    saja: negatif acak murni akan menumpuk di periode maksimum siklus, karena di
    situ jumlah AR paling banyak. CNN lalu bisa belajar membedakan "tahun ramai"
    dari "tahun sepi" — bukan membedakan AR yang akan flare.

    Subsampling ini HANYA mengubah komposisi data latih, jadi basis rate yang
    dilihat model bukan lagi basis rate sebenarnya. Itu sebabnya ambang
    keputusan tetap wajib dicari pada set validasi yang tidak di-subsample.
    """
    rng = np.random.default_rng(cfg.random_state if random_state is None else random_state)
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]

    n_neg = min(len(neg), neg_per_pos * len(pos))
    if n_neg < len(neg):
        bulan = neg["T_REC"].dt.to_period("M")
        jatah = (bulan.value_counts(normalize=True) * n_neg).round().astype(int)
        pilihan = []
        for periode, k in jatah.items():
            blok = neg[bulan == periode]
            k = min(int(k), len(blok))
            if k > 0:
                pilihan.append(blok.iloc[rng.choice(len(blok), k, replace=False)])
        neg = pd.concat(pilihan) if pilihan else neg.iloc[:0]

    out = pd.concat([pos, neg]).sort_values("T_REC")
    if max_rows is not None and len(out) > max_rows:
        out = out.iloc[np.sort(rng.choice(len(out), max_rows, replace=False))]

    print(f"[CNN] Subset citra: {len(out):,} baris "
          f"({int(out['y'].sum()):,} positif, {int((out['y'] == 0).sum()):,} negatif) "
          f"~{len(out) * 0.6:.0f} MB unduhan.")
    return out.reset_index(drop=True)


def select_ar_blocks(df: pd.DataFrame, cfg: Config, n_ars: int = 40,
                     max_frames_per_ar: int = 24, pos_ar_fraction: float = 0.6,
                     random_state: int | None = None) -> pd.DataFrame:
    """Pilih beberapa AR utuh, masing-masing satu potongan waktu KONTIGU.

    Alternatif dari `select_patch_subset`, dan untuk unduhan berskala besar
    hampir selalu pilihan yang benar. Alasannya operasional sekaligus ilmiah.

    Operasional: JSOC hanya mengizinkan satu permintaan ekspor tertunda per
    akun, dan tiap permintaan memakan beberapa menit. Yang menentukan lama
    unduhan karena itu adalah JUMLAH PERMINTAAN, bukan jumlah berkas. 800 frame
    yang tersebar di 330 AR butuh ratusan permintaan; 800 frame dari 40 AR
    yang masing-masing kontigu butuh 40 — dua orde besaran lebih cepat.

    Ilmiah: frame yang berurutan memperlihatkan EVOLUSI AR. Snapshot tunggal
    tidak bisa membedakan AR yang sedang tumbuh dari yang sedang meluruh,
    padahal peluang flare keduanya berbeda. Potongan kontigu juga prasyarat
    kalau nanti naik ke ConvLSTM.

    Harganya jujur: jumlah AR yang dilihat model jadi jauh lebih sedikit, dan
    keragaman itulah yang paling menentukan generalisasi. Jadi ini pertukaran,
    bukan perbaikan gratis — naikkan `n_ars` sebesar yang waktu Anda izinkan,
    dan kecilkan `max_frames_per_ar` lebih dulu kalau harus memilih.
    """
    rng = np.random.default_rng(cfg.random_state if random_state is None else random_state)

    per_ar = df.groupby("HARPNUM")["y"].max()
    ar_pos = per_ar[per_ar == 1].index.to_numpy()
    ar_neg = per_ar[per_ar == 0].index.to_numpy()

    n_pos = min(len(ar_pos), int(round(n_ars * pos_ar_fraction)))
    n_neg = min(len(ar_neg), n_ars - n_pos)
    pilih = np.concatenate([rng.choice(ar_pos, n_pos, replace=False),
                            rng.choice(ar_neg, n_neg, replace=False)])

    potongan = []
    for h in pilih:
        g = df[df["HARPNUM"] == h].sort_values("T_REC")
        if len(g) <= max_frames_per_ar:
            potongan.append(g)
            continue
        # Pusatkan jendela pada kejadian positif pertama kalau ada — di situlah
        # evolusi yang relevan terjadi. Kalau tidak ada, ambil bagian tengah,
        # karena tepi rekaman HARP biasanya AR yang baru muncul atau sudah lewat limb.
        pos = np.where(g["y"].to_numpy() == 1)[0]
        pusat = int(pos[0]) if len(pos) else len(g) // 2
        awal = int(np.clip(pusat - max_frames_per_ar // 2, 0, len(g) - max_frames_per_ar))
        potongan.append(g.iloc[awal:awal + max_frames_per_ar])

    out = pd.concat(potongan).sort_values("T_REC").reset_index(drop=True)
    print(f"[CNN] Blok AR: {len(out):,} frame dari {len(pilih)} AR "
          f"({n_pos} pernah flare, {n_neg} tidak) | "
          f"{int(out['y'].sum()):,} frame positif | ~{len(out) * 0.6:.0f} MB unduhan.")
    print(f"[CNN] Perkiraan {len(pilih)} permintaan ekspor JSOC "
          f"(vs ~{out[['HARPNUM', 'T_REC']].drop_duplicates().shape[0]} "
          f"kalau frame-nya tersebar).")
    return out


def missing_patches(df: pd.DataFrame, cfg: Config,
                    fits_dir: Path | None = None) -> pd.DataFrame:
    """Baris yang berkas FITS-nya belum ada di disk."""
    fits_dir = Path(fits_dir or cfg.fits_dir)
    rows = df[["HARPNUM", "T_REC"]].drop_duplicates()
    if rows.empty:
        return rows
    ada = rows.apply(
        lambda r: (fits_dir / patch_filename(r["HARPNUM"], r["T_REC"])).exists(), axis=1)
    return rows[~ada]


# ==============================================================================
# 2. Cache piksel — FITS -> satu berkas biner float16
# ==============================================================================

@dataclass
class PatchCache:
    """Array (N, H, W) float16 di disk plus indeks (HARPNUM, T_REC) -> baris.

    Disimpan sebagai raw binary, bukan `.npy`, supaya bisa DITAMBAH tanpa
    menulis ulang seluruh berkas: menambah 5.000 citra ke cache 100.000 citra
    hanya menempelkan byte di ujungnya.
    """

    path: Path
    index: pd.DataFrame        # kolom: HARPNUM, T_REC, row
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.index), self.height, self.width)

    def open_memmap(self) -> np.memmap:
        return np.memmap(self.path, dtype=np.float16, mode="r", shape=self.shape)

    def lookup(self, df: pd.DataFrame) -> np.ndarray:
        """Posisi baris cache untuk tiap baris `df`; -1 bila citranya tidak ada."""
        kunci = (self.index.drop_duplicates(subset=["HARPNUM", "T_REC"])
                 .set_index(["HARPNUM", "T_REC"])["row"])
        minta = pd.MultiIndex.from_arrays([df["HARPNUM"].astype("int64"),
                                           pd.to_datetime(df["T_REC"])])
        return kunci.reindex(minta).fillna(-1).to_numpy(dtype=np.int64)

    def available_mask(self, df: pd.DataFrame) -> np.ndarray:
        return self.lookup(df) >= 0


def cache_paths(cfg: Config, size: tuple[int, int]) -> tuple[Path, Path]:
    stem = f"patches_{size[0]}x{size[1]}"
    return cfg.processed_dir / f"{stem}.f16", cfg.processed_dir / f"{stem}_index.parquet"


def build_patch_cache(df: pd.DataFrame, cfg: Config,
                      size: tuple[int, int] = DEFAULT_SIZE,
                      fits_dir: Path | None = None,
                      rebuild: bool = False, verbose_every: int = 200) -> PatchCache:
    """Bangun (atau tambah) cache piksel untuk baris-baris `df`.

    Aman dipanggil berulang: baris yang sudah ada di indeks dilewati, jadi
    menambah data baru tidak perlu mengulang dekode FITS dari nol.
    """
    cfg.ensure_dirs()
    bin_path, idx_path = cache_paths(cfg, size)
    fits_dir = Path(fits_dir or cfg.fits_dir)
    H, W = size

    if rebuild:
        bin_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)

    if idx_path.exists() and bin_path.exists():
        index = pd.read_parquet(idx_path)
        index["T_REC"] = pd.to_datetime(index["T_REC"])
    else:
        index = pd.DataFrame({"HARPNUM": pd.Series(dtype="int64"),
                              "T_REC": pd.Series(dtype="datetime64[ns]"),
                              "row": pd.Series(dtype="int64")})

    # Indeks dan berkas biner harus konsisten. Kalau salah satu hilang (run
    # sebelumnya terputus di tengah), menambah data akan menulis di belakang
    # byte lama sementara indeks menghitung dari nol — seluruh pemetaan baris
    # jadi salah TANPA satu pun pesan error, dan CNN akan dilatih pada pasangan
    # citra-label yang tidak nyambung. Lebih baik buang dan bangun ulang.
    harap = len(index) * H * W * 2       # float16 = 2 byte
    nyata = bin_path.stat().st_size if bin_path.exists() else 0
    if nyata != harap:
        if nyata:
            print(f"[CNN] Cache tidak konsisten ({nyata:,} byte, seharusnya "
                  f"{harap:,}); dibangun ulang dari nol.")
        bin_path.unlink(missing_ok=True)
        index = index.iloc[:0]

    sudah = set(zip(index["HARPNUM"], index["T_REC"]))
    perlu = (df[["HARPNUM", "T_REC"]].drop_duplicates()
             .assign(HARPNUM=lambda d: d["HARPNUM"].astype("int64"),
                     T_REC=lambda d: pd.to_datetime(d["T_REC"])))
    perlu = perlu[[k not in sudah for k in zip(perlu["HARPNUM"], perlu["T_REC"])]]

    if perlu.empty:
        print(f"[CNN] Cache piksel sudah lengkap: {len(index):,} citra "
              f"@ {H}x{W} ({bin_path.name}).")
        return PatchCache(bin_path, index.reset_index(drop=True), H, W)

    baris_baru, gagal, hilang = [], 0, 0
    next_row = int(index["row"].max()) + 1 if len(index) else 0

    print(f"[CNN] Menyiapkan {len(perlu):,} citra baru -> {bin_path.name} @ {H}x{W}")
    with open(bin_path, "ab") as fh:
        for h, t in zip(perlu["HARPNUM"], perlu["T_REC"]):
            p = fits_dir / patch_filename(h, t)
            if not p.exists():
                hilang += 1
                continue
            try:
                arr = normalize_bz(resize_patch(read_patch(p), size))
            except Exception as exc:  # noqa: BLE001
                print(f"[CNN] gagal membaca {p.name}: {exc}")
                gagal += 1
                continue
            fh.write(arr.astype(np.float16).tobytes())
            baris_baru.append({"HARPNUM": int(h), "T_REC": t, "row": next_row})
            next_row += 1
            if verbose_every and len(baris_baru) % verbose_every == 0:
                print(f"[CNN]   {len(baris_baru):,}/{len(perlu):,} citra diproses")

    if baris_baru:
        index = pd.concat([index, pd.DataFrame(baris_baru)], ignore_index=True)
        index.to_parquet(idx_path, index=False)

    ukuran_mb = bin_path.stat().st_size / 1e6 if bin_path.exists() else 0.0
    print(f"[CNN] Cache: {len(index):,} citra ({ukuran_mb:.1f} MB) | "
          f"+{len(baris_baru):,} baru, {hilang:,} FITS belum diunduh, "
          f"{gagal:,} gagal dibaca.")
    if hilang:
        print("[CNN] Unduh sisanya lewat `magnetogram.download_patches(...)` "
              "atau `run_pipeline.py --cnn --cnn-download`.")
    return PatchCache(bin_path, index.reset_index(drop=True), H, W)


def load_patch_cache(cfg: Config, size: tuple[int, int] = DEFAULT_SIZE) -> PatchCache | None:
    """Muat cache yang sudah ada tanpa menyentuh FITS. None bila belum dibangun."""
    bin_path, idx_path = cache_paths(cfg, size)
    if not (bin_path.exists() and idx_path.exists()):
        return None
    index = pd.read_parquet(idx_path)
    index["T_REC"] = pd.to_datetime(index["T_REC"])
    return PatchCache(bin_path, index, size[0], size[1])


def restrict_to_cache(splits: dict[str, pd.DataFrame],
                      cache: PatchCache) -> dict[str, pd.DataFrame]:
    """Sisakan hanya baris yang punya citra, di SEMUA split sekaligus.

    Ini bukan detail teknis melainkan syarat keabsahan perbandingan: kalau CNN
    dinilai pada 3.000 baris test sementara Random Forest dinilai pada 9.000
    baris test yang komposisinya berbeda, kedua angka TSS itu tidak bisa
    disandingkan sama sekali. Seluruh tangga model harus melihat baris yang
    persis sama.
    """
    out = {}
    print("\n[CNN] Membatasi seluruh tangga model ke baris yang punya citra:")
    for nama, part in splits.items():
        if len(part) == 0:
            out[nama] = part
            continue
        sisa = part[cache.available_mask(part)]
        out[nama] = sisa
        info = (f"positif {sisa['y'].mean():.2%}, {sisa['HARPNUM'].nunique()} HARP"
                if len(sisa) else "KOSONG")
        print(f"      {nama:<6} {len(part):>7,} -> {len(sisa):>7,} sampel | {info}")
    print()
    return out


# ==============================================================================
# 3. Arsitektur
# ==============================================================================

def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Tingkat 4 (CNN) membutuhkan PyTorch, yang sengaja tidak masuk "
            "requirements dasar karena ukurannya ~2 GB.\n\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n\n"
            "Tingkat 0-3 tetap berjalan tanpa PyTorch."
        ) from exc
    return torch


def build_network(dropout: float = 0.3, n_scalar: int = 0, width: int = 16):
    """CNN kecil: 4 blok konvolusi, pooling global, kepala linear.

    Semua pilihan di sini sengaja konservatif karena datanya sedikit:

    * ~10^5 parameter, bukan 10^7. ResNet-50 di sini akan menghafal set latih
      dalam beberapa epoch dan tidak menggeneralisasi sama sekali.
    * BatchNorm setelah tiap konvolusi — rentang dinamis masukan tetap lebar
      meski sudah dikompresi arcsinh.
    * Pooling global rata-rata DAN maksimum lalu digabung. Rata-rata menangkap
      "seberapa kuat AR ini secara keseluruhan", maksimum menangkap "adakah satu
      titik yang sangat tajam" (PIL curam); keduanya berbeda secara fisis.
      Pooling global juga membuat jaringan tahan terhadap pergeseran posisi AR
      di dalam bounding box.
    * Fusi fitur skalar opsional: keyword SHARP digabungkan di kepala
      klasifikasi, sehingga CNN hanya perlu mempelajari SISA informasi yang
      belum terwakili di sana.
    """
    torch = _require_torch()
    nn = torch.nn

    def blok(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    w = width

    class MagnetogramCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fitur = nn.Sequential(blok(1, w), blok(w, 2 * w),
                                       blok(2 * w, 4 * w), blok(4 * w, 8 * w))
            self.n_scalar = n_scalar
            dim = 8 * w * 2 + n_scalar          # avg-pool + max-pool (+ skalar)
            self.kepala = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim, 64), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x, s=None):
            f = self.fitur(x)
            v = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
            if self.n_scalar:
                v = torch.cat([v, s], dim=1)
            return self.kepala(v).squeeze(1)

    return MagnetogramCNN()


# ==============================================================================
# 4. Pembungkus bergaya scikit-learn
# ==============================================================================

class CNNClassifier:
    """CNN magnetogram dengan antarmuka `fit` / `predict_proba`.

    Sengaja meniru API scikit-learn supaya `run_pipeline.finish()`,
    `ProbabilityCalibrator`, dan seluruh `evaluate.py` bisa memperlakukan
    tingkat 4 persis seperti tingkat 2 dan 3 — tanpa satu pun cabang khusus.

    `X` yang diterima adalah DataFrame yang WAJIB memuat kolom `HARPNUM` dan
    `T_REC` (kunci pencarian citra di cache), dan boleh memuat kolom fitur
    skalar bila `scalar_features` diisi.

    Tiga hal yang menentukan apakah model ini masuk akal atau tidak:

    Ketidakseimbangan kelas
        `pos_weight` pada BCE, bukan oversampling. Oversampling menduplikasi
        citra positif yang di data ini nyaris identik antar-jam, sehingga yang
        bertambah cuma peluang menghafal, bukan informasi.

    Early stopping
        Dipantau pada potongan KRONOLOGIS TERAKHIR dari set latih, bukan
        potongan acak — sampel AR yang sama pada jam berdekatan nyaris identik,
        jadi validasi acak akan bocor dan memberi sinyal berhenti yang terlalu
        optimistis. Potongan ini juga berbeda dari set validasi utama, yang
        sudah dipakai untuk kalibrasi dan pemilihan ambang.

    Augmentasi
        Cerminan kiri-kanan dan atas-bawah. Keduanya menghasilkan konfigurasi
        medan yang tetap fisis mungkin. Catatan jujur: aturan hemisfer
        (kecenderungan tanda heliisitas berbeda di utara dan selatan) sedikit
        dilanggar oleh cermin vertikal — harganya kecil dibanding manfaatnya
        pada data sesedikit ini, tapi matikan lewat `augment=False` bila Anda
        ingin membandingkan pengaruhnya.
    """

    name = "4. CNN magnetogram"

    def __init__(self, cache: PatchCache, scalar_features: list[str] | None = None,
                 epochs: int = 40, batch_size: int = 64, lr: float = 1e-3,
                 weight_decay: float = 1e-4, dropout: float = 0.3,
                 width: int = 16, patience: int = 8, monitor: str = "AP",
                 val_frac: float = 0.15, augment: bool = True,
                 device: str | None = None, random_state: int = 42,
                 verbose: bool = True):
        self.cache = cache
        self.scalar_features = list(scalar_features or [])
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.width = width
        self.patience = patience
        self.monitor = monitor
        self.val_frac = val_frac
        self.augment = augment
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ util --
    def _rows(self, X: pd.DataFrame) -> np.ndarray:
        for kol in ("HARPNUM", "T_REC"):
            if kol not in X.columns:
                raise KeyError(
                    f"CNNClassifier butuh kolom '{kol}' untuk mencari citra di cache. "
                    "Sertakan HARPNUM dan T_REC pada X (lihat `cnn_columns()`)."
                )
        return self.cache.lookup(X)

    def _scalars(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray | None:
        if not self.scalar_features:
            return None
        from sklearn.preprocessing import StandardScaler

        from .models import SignedLog

        V = np.nan_to_num(np.asarray(X[self.scalar_features], dtype=float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        V = SignedLog().transform(V)
        if fit:
            self._scaler_ = StandardScaler().fit(V)
        return self._scaler_.transform(V).astype(np.float32)

    def _batch_images(self, rows: np.ndarray) -> np.ndarray:
        """Ambil citra dari memmap. Baris tanpa citra diisi nol."""
        H, W = self.cache.height, self.cache.width
        out = np.zeros((len(rows), H, W), dtype=np.float32)
        ada = rows >= 0
        if ada.any():
            urut = np.argsort(rows[ada])           # akses memmap terurut = jauh lebih cepat
            idx = np.where(ada)[0][urut]
            out[idx] = np.asarray(self._mm_[rows[idx]], dtype=np.float32)
        return out

    def _augment(self, imgs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, H, W = imgs.shape
        flip_h = rng.random(n) < 0.5
        flip_v = rng.random(n) < 0.5
        imgs[flip_h] = imgs[flip_h][:, :, ::-1]
        imgs[flip_v] = imgs[flip_v][:, ::-1, :]

        # Geseran kecil dengan isian nol (bukan wrap-around, yang akan
        # memindahkan tepi AR ke sisi berlawanan dan menciptakan PIL palsu).
        maks = max(2, H // 32)
        dy, dx = rng.integers(-maks, maks + 1, size=2)
        if dy or dx:
            geser = np.zeros_like(imgs)
            ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
            xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
            geser[:, yd, xd] = imgs[:, ys, xs]
            imgs = geser
        return np.ascontiguousarray(imgs)

    def _score(self, y_true: np.ndarray, p: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        if self.monitor == "AP":
            return float(average_precision_score(y_true, p))
        if self.monitor == "ROC_AUC":
            return float(roc_auc_score(y_true, p))
        if self.monitor == "TSS":
            from .evaluate import best_threshold
            return float(best_threshold(y_true, p, "TSS")[1])
        raise ValueError(f"monitor tidak dikenal: {self.monitor}")

    # ------------------------------------------------------------------- fit --
    def fit(self, X: pd.DataFrame, y):
        torch = _require_torch()

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._mm_ = self.cache.open_memmap()

        X = X.reset_index(drop=True)
        y = np.asarray(y).astype(np.float32)
        rows = self._rows(X)
        n_hilang = int((rows < 0).sum())
        if n_hilang:
            print(f"[CNN] {n_hilang:,} baris latih tidak punya citra dan dibuang. "
                  f"Gunakan `restrict_to_cache()` agar perbandingan antar-model adil.")
            simpan = rows >= 0
            X, y, rows = X[simpan].reset_index(drop=True), y[simpan], rows[simpan]
        if len(X) == 0:
            raise RuntimeError(
                "Tidak ada satu pun citra untuk set latih. Bangun cache dulu: "
                "`build_patch_cache(df, cfg)` setelah mengunduh FITS-nya."
            )

        S = self._scalars(X, fit=True)

        # Pisah kronologis: ekor waktu terakhir dipakai untuk early stopping.
        urut = np.argsort(X["T_REC"].to_numpy())
        potong = int(len(urut) * (1 - self.val_frac))
        i_tr, i_va = urut[:potong], urut[potong:]
        if len(i_va) == 0 or y[i_va].sum() == 0:
            # Tanpa positif di ekor, skor pemantauan tidak terdefinisi;
            # lebih jujur berhenti pada jumlah epoch tetap daripada memantau
            # angka yang tidak bermakna.
            i_tr, i_va = urut, np.array([], dtype=int)

        n_pos, n_neg = float(y[i_tr].sum()), float(len(i_tr) - y[i_tr].sum())
        pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)],
                                  dtype=torch.float32, device=dev)

        self.model_ = build_network(self.dropout, len(self.scalar_features),
                                    self.width).to(dev)
        n_par = sum(p.numel() for p in self.model_.parameters())
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if self.verbose:
            print(f"\n[CNN] {n_par:,} parameter | perangkat {dev} | "
                  f"latih {len(i_tr):,} ({int(y[i_tr].sum()):,} positif) | "
                  f"pantau {len(i_va):,} sampel [{self.monitor}] | "
                  f"pos_weight {pos_weight.item():.1f}")
            if y[i_tr].sum() < 100:
                print("[CNN] PERINGATAN: kurang dari 100 sampel positif di set latih. "
                      "Hasil CNN pada skala ini nyaris pasti didominasi overfitting; "
                      "perbesar rentang data sebelum menyimpulkan apa pun.")

        terbaik, sabar, bobot_terbaik = -np.inf, 0, None
        self.history_ = []

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            acak = rng.permutation(len(i_tr))
            total, n_batch = 0.0, 0
            for b in range(0, len(acak), self.batch_size):
                sel = i_tr[acak[b:b + self.batch_size]]
                imgs = self._batch_images(rows[sel])
                if self.augment:
                    imgs = self._augment(imgs, rng)
                xb = torch.from_numpy(imgs).unsqueeze(1).to(dev)
                yb = torch.from_numpy(y[sel]).to(dev)
                sb = torch.from_numpy(S[sel]).to(dev) if S is not None else None

                opt.zero_grad(set_to_none=True)
                loss = lossf(self.model_(xb, sb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()
                total += float(loss.item())
                n_batch += 1
            sched.step()

            train_loss = total / max(n_batch, 1)
            if len(i_va):
                p_va = self._forward_rows(rows[i_va], S[i_va] if S is not None else None, dev)
                skor = self._score(y[i_va], p_va)
            else:
                skor = -train_loss     # tanpa set pantau: pakai loss sebagai proksi
            self.history_.append({"epoch": epoch, "loss": train_loss, self.monitor: skor})

            if self.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == self.epochs):
                print(f"[CNN]   epoch {epoch:>3}/{self.epochs}  loss {train_loss:.4f}  "
                      f"{self.monitor} {skor:+.4f}")

            if skor > terbaik + 1e-6:
                terbaik, sabar = skor, 0
                bobot_terbaik = {k: v.detach().clone()
                                 for k, v in self.model_.state_dict().items()}
            else:
                sabar += 1
                if sabar >= self.patience:
                    if self.verbose:
                        print(f"[CNN]   early stopping di epoch {epoch} "
                              f"({self.monitor} terbaik {terbaik:+.4f})")
                    break

        if bobot_terbaik is not None:
            self.model_.load_state_dict(bobot_terbaik)
        self.best_score_ = float(terbaik)
        self.device_ = dev
        return self

    # --------------------------------------------------------------- predict --
    def _forward_rows(self, rows: np.ndarray, S: np.ndarray | None, dev) -> np.ndarray:
        torch = _require_torch()
        self.model_.eval()
        keluar = np.empty(len(rows), dtype=np.float64)
        with torch.no_grad():
            for b in range(0, len(rows), self.batch_size):
                sl = slice(b, b + self.batch_size)
                xb = torch.from_numpy(self._batch_images(rows[sl])).unsqueeze(1).to(dev)
                sb = torch.from_numpy(S[sl]).to(dev) if S is not None else None
                keluar[sl] = torch.sigmoid(self.model_(xb, sb)).cpu().numpy()
        return keluar

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Panggil fit() dulu.")
        X = X.reset_index(drop=True)
        if not hasattr(self, "_mm_"):
            self._mm_ = self.cache.open_memmap()
        rows = self._rows(X)
        hilang = int((rows < 0).sum())
        if hilang:
            print(f"[CNN] {hilang:,} dari {len(X):,} baris tidak punya citra; "
                  f"diprediksi dari citra nol (skornya tidak bermakna).")
        p = self._forward_rows(rows, self._scalars(X), self.device_)
        return np.column_stack([1 - p, p])

    # --------------------------------------------------------------- laporan --
    def describe(self) -> str:
        skalar = (f", + {len(self.scalar_features)} fitur SHARP di kepala"
                  if self.scalar_features else "")
        return (f"CNN {self.cache.height}x{self.cache.width} piksel{skalar}; "
                f"{self.monitor} pantau terbaik {getattr(self, 'best_score_', float('nan')):+.4f}")

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(getattr(self, "history_", []))


def cnn_columns(feats: list[str], scalar_features: list[str] | None = None) -> list[str]:
    """Kolom yang harus ikut diserahkan ke CNNClassifier: kunci citra + skalar."""
    return list(dict.fromkeys(["HARPNUM", "T_REC"] + list(scalar_features or [])))


def plot_training_curve(model: CNNClassifier, out_path: Path):
    """Kurva loss latih dan skor pantau per epoch — pemeriksaan overfitting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = model.history_frame()
    if hist.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(hist["epoch"], hist["loss"], color="#c0392b", label="loss latih")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss latih (BCE terbobot)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist[model.monitor], color="#2471a3",
             label=f"{model.monitor} pantau")
    ax2.set_ylabel(f"{model.monitor} (potongan pantau)", color="#2471a3")
    ax2.tick_params(axis="y", labelcolor="#2471a3")

    ax1.set_title("Tingkat 4 — kurva latih CNN magnetogram")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
