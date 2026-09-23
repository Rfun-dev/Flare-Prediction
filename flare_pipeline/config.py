"""Konfigurasi terpusat untuk pipeline prediksi flare berbasis magnetogram SDO/HMI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Direktori root proyek (satu level di atas package ini)
ROOT = Path(__file__).resolve().parent.parent

# Versi logika pelabelan; ikut masuk ke nama cache dataset berlabel.
LABEL_VERSION = 2


# ==============================================================================
# Daftar keyword SHARP.
# Ini adalah *fitur turunan magnetogram*: setiap keyword dihitung JSOC langsung
# dari peta vektor medan magnet (magnetogram) di dalam bounding box HARP.
# Subset ini mengikuti Bobra & Couvidat (2015), ApJ 798, 135 — feature set
# standar de-facto untuk flare forecasting berbasis SHARP.
# ==============================================================================
# Catatan: TOTBSQ (total magnitude of Lorentz force) sengaja TIDAK disertakan.
# Keyword itu ada di `hmi.sharp_720s` tapi TIDAK ada di seri CEA yang dipakai di
# sini; JSOC tetap mengembalikan kolomnya sebagai NaN penuh tanpa melempar
# error, sehingga `dropna` akan menghapus seluruh dataset secara senyap.
SHARP_FEATURES: tuple[str, ...] = (
    "TOTUSJH",    # total unsigned current helicity
    "TOTPOT",     # total photospheric magnetic free energy density
    "TOTUSJZ",    # total unsigned vertical current
    "ABSNJZH",    # absolute value of net current helicity
    "SAVNCPP",    # sum of the modulus of the net current per polarity
    "USFLUX",     # total unsigned flux
    "AREA_ACR",   # area of strong-field pixels in the active region
    "MEANPOT",    # mean photospheric magnetic free energy
    "R_VALUE",    # sum of flux near polarity inversion line (Schrijver R)
    "SHRGT45",    # fraction of area with shear angle > 45 deg
    "MEANSHR",    # mean shear angle
    "MEANGAM",    # mean angle of field from radial
    "MEANGBT",    # mean gradient of total field
    "MEANGBZ",    # mean gradient of vertical field
    "MEANGBH",    # mean gradient of horizontal field
    "MEANJZH",    # mean current helicity (Bz contribution)
    "MEANJZD",    # mean vertical current density
    "MEANALP",    # mean characteristic twist parameter alpha
)

# Keyword non-fitur yang tetap harus diambil (identitas, waktu, quality control)
SHARP_METADATA_KEYS: tuple[str, ...] = (
    "T_REC",      # waktu observasi (TAI)
    "HARPNUM",    # id HARP
    "NOAA_AR",    # nomor NOAA AR utama yang terkait
    "NOAA_ARS",   # daftar semua NOAA AR di dalam HARP (dipisah koma)
    "NOAA_NUM",   # jumlah NOAA AR di dalam HARP
    "LON_FWT",    # bujur Carrington terboboti fluks -> proksi CMD
    "LAT_FWT",    # lintang terboboti fluks
    "OBS_VR",     # kecepatan radial SDO relatif Matahari (m/s)
    "QUALITY",    # bitmask kualitas data (0 = bersih)
    "CRLN_OBS",   # bujur Carrington sub-satelit (untuk hitung CMD sejati)
)


@dataclass
class Config:
    # ---------------------------------------------------------------- data ---
    start_time: str = "2010-05-01"   # awal era HMI
    # Mencakup siklus 24 DAN siklus 25. Siklus 25 memuncak sekitar 2024 dan
    # jauh lebih produktif daripada 2017-2018: Juni 2024 tercatat 320 flare GOES
    # sementara Juni 2019 hanya 4. Menghentikan rentang di 2018 berarti membuang
    # justru periode yang paling kaya kejadian positif.
    end_time: str = "2025-12-31"

    jsoc_email: str = "erfan.ferdianto2003@gmail.com"
    sharp_series: str = "hmi.sharp_cea_720s"

    # Cadence pengambilan sampel SHARP. 1h adalah kompromi umum:
    # cukup rapat untuk time-series, tidak membanjiri JSOC.
    #
    # PERINGATAN: JANGAN pakai cadence 6h atau 12h. SDO menjalankan manuver
    # kalibrasi harian pada 06:00 dan 18:00 TAI, dan pada kedua jam itu
    # QUALITY selalu != 0 (terukur: 100% baris pada jam tsb). Cadence yang
    # membagi habis 6 jam akan kehilangan separuh data secara sistematis
    # setelah filter QUALITY. Gunakan 1h, 2h, atau 4h.
    sampling_cadence_hours: int = 1

    # -------------------------------------------------------------- windowing ---
    forecast_horizon_hours: int = 24   # prediksi flare dalam T+24 jam
    obs_window_hours: int = 24         # riwayat yang dipakai persistence / fitur time-series
    # Jeda antara akhir window fitur dan awal window prediksi.
    # 0 = prediksi mulai tepat saat pengamatan (skema standar Bobra & Couvidat).
    latency_hours: int = 0

    # --------------------------------------------------------- quality control ---
    cmd_max_deg: float = 70.0            # buang AR dekat limb (proyeksi terdistorsi)
    radial_velocity_max: float = 3500.0  # m/s, buang periode orbital velocity ekstrem
    require_quality_zero: bool = True     # hanya terima QUALITY == 0

    # -------------------------------------------------------------- pelabelan ---
    threshold_class: str = "M"   # positif jika flare >= M1.0 dalam horizon

    # ------------------------------------------------------------------ split ---
    # Split KRONOLOGIS (bukan acak) untuk mencegah kebocoran temporal:
    # sampel AR yang sama pada jam berdekatan hampir identik.
    #
    # Batasnya sengaja ditaruh agar set uji jatuh di FASE NAIK SIKLUS 25
    # (2022-2025), bukan di minimum. Dengan batas lama (uji 2017-2018) seluruh
    # kejadian positif di set uji berasal dari 3 HARP saja, sehingga bootstrap
    # per HARP kehabisan blok positif dan selang TSS melebar sampai ~0,99.
    train_end: str = "2019-12-31"   # siklus 24 + minimum
    val_end: str = "2021-12-31"     # awal naiknya siklus 25
    # sisanya -> test (2022-2025, fase naik + maksimum siklus 25)

    # ----------------------------------------------------------------- output ---
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    output_dir: Path = field(default_factory=lambda: ROOT / "output_dataset")
    fits_dir: Path = field(default_factory=lambda: ROOT / "magnetogram_fits")

    random_state: int = 42

    # -------------------------------------------------------------- turunan ---
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def sharp_keys(self) -> tuple[str, ...]:
        return SHARP_METADATA_KEYS + SHARP_FEATURES

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.output_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    def dataset_signature(self) -> str:
        """Sidik jari seluruh parameter yang MEMPENGARUHI isi dataset berlabel.

        Dipakai sebagai nama berkas cache. Tanpa ini, mengubah misalnya
        `forecast_horizon_hours` lalu menjalankan `--offline` akan diam-diam
        memakai label lama dari horizon sebelumnya — hasilnya salah tanpa
        satu pun pesan peringatan.
        """
        t0 = pd.Timestamp(self.start_time).strftime("%Y%m%d")
        t1 = pd.Timestamp(self.end_time).strftime("%Y%m%d")
        return (f"{t0}-{t1}"
                f"_cad{self.sampling_cadence_hours}h"
                f"_obs{self.obs_window_hours}h"
                f"_lat{self.latency_hours}h"
                f"_fcst{self.forecast_horizon_hours}h"
                f"_{self.threshold_class}"
                f"_cmd{self.cmd_max_deg:g}"
                f"_vr{self.radial_velocity_max:g}"
                f"_q{int(self.require_quality_zero)}"
                # Versi logika pelabelan. Naikkan bila cara flare dijodohkan
                # ke AR berubah, supaya cache berlabel lama tidak terpakai.
                # v2: nomor AR yang hilang di katalog SWPC ditambal dari SSW.
                f"_lab{LABEL_VERSION}")

    def labeled_dataset_path(self) -> Path:
        return self.processed_dir / f"labeled_{self.dataset_signature()}.parquet"


# Preset cepat: 2 tahun data untuk uji coba pipeline tanpa menunggu lama.
# Cadence 4h (bukan 6h) agar tidak selalu jatuh di jam kalibrasi 06/18 TAI.
def quick_config(**overrides) -> Config:
    cfg = Config(
        start_time="2013-01-01",
        end_time="2014-12-31",
        sampling_cadence_hours=4,
        train_end="2013-12-31",
        val_end="2014-06-30",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


CFG = Config()
