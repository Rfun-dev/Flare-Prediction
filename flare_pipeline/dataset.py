"""Quality control, windowing, pelabelan, rekayasa fitur, dan pembagian data."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .config import SHARP_FEATURES, Config
from .labeling import (
    FlareIndex,
    flux_to_binary_label,
    flux_to_multiclass_label,
    goes_class_to_flux,
)


# ==============================================================================
# 1. Quality control
# ==============================================================================

def compute_cmd(df: pd.DataFrame) -> pd.Series:
    """Central Meridian Distance dalam derajat.

    Diverifikasi lewat `drms.Client.info('hmi.sharp_cea_720s')`: LON_FWT adalah
    bujur *Stonyhurst* ("Stonyhurst longitude of flux-weighted center of active
    pixels"), yang menurut definisinya diukur dari meridian tengah — jadi
    LON_FWT SUDAH merupakan CMD. Jangan dikurangi CRLN_OBS (itu bujur
    Carrington pengamat; menguranginya menghasilkan nilai di balik limb).
    """
    return ((df["LON_FWT"] + 180.0) % 360.0) - 180.0


def quality_control(df_sharp: pd.DataFrame, cfg: Config,
                    verbose: bool = True) -> pd.DataFrame:
    """Terapkan filter kualitas standar literatur SHARP."""
    df = df_sharp.copy()
    steps: list[tuple[str, int]] = [("awal", len(df))]

    if cfg.require_quality_zero and "QUALITY" in df.columns:
        df = df[df["QUALITY"].fillna(-1) == 0]
        steps.append(("QUALITY == 0", len(df)))

    if "LON_FWT" in df.columns:
        df = df[df["LON_FWT"].notna()]
        df["CMD_DEG"] = compute_cmd(df)
        df = df[df["CMD_DEG"].abs() <= cfg.cmd_max_deg]
        steps.append((f"|CMD| <= {cfg.cmd_max_deg}deg", len(df)))

    if "OBS_VR" in df.columns:
        df = df[df["OBS_VR"].abs() <= cfg.radial_velocity_max]
        steps.append((f"|OBS_VR| <= {cfg.radial_velocity_max:.0f} m/s", len(df)))

    feats = [c for c in SHARP_FEATURES if c in df.columns]
    # Penjaga: kolom yang NaN 100% berarti keyword tidak ada di seri ini. JSOC
    # tidak melempar error untuk keyword tak dikenal — ia mengembalikan kolom
    # kosong — sehingga tanpa penjaga ini `dropna` di bawah akan menghapus
    # SELURUH dataset dan penyebabnya baru terlihat jauh di hilir.
    empty = [c for c in feats if df[c].isna().all()]
    if empty:
        print(f"[QC] PERINGATAN: {empty} kosong 100% — kemungkinan keyword tidak "
              f"tersedia di seri '{cfg.sharp_series}'. Kolom ini diabaikan.")
        feats = [c for c in feats if c not in empty]
    df = df.dropna(subset=feats)
    steps.append(("fitur SHARP lengkap", len(df)))

    # HARP tanpa nomor NOAA tidak bisa dicocokkan ke katalog flare.
    df["NOAA_AR"] = pd.to_numeric(df.get("NOAA_AR"), errors="coerce").fillna(0).astype(int)
    has_ars = df.get("NOAA_ARS", pd.Series("", index=df.index)).astype(str).str.strip()
    df = df[(df["NOAA_AR"] != 0) | has_ars.str.contains(r"\d")]
    steps.append(("punya NOAA AR", len(df)))

    if verbose:
        print("\n[QC] Jejak penyaringan:")
        prev = steps[0][1]
        for name, n in steps:
            drop = prev - n
            print(f"      {name:<28} {n:>8,}" + (f"   (-{drop:,})" if drop else ""))
            prev = n
        print()

    return df.sort_values(["HARPNUM", "T_REC"]).reset_index(drop=True)


def _ar_list(row) -> list[int]:
    """Semua NOAA AR yang termuat dalam satu HARP (bisa lebih dari satu)."""
    ars: list[int] = []
    raw = str(row.get("NOAA_ARS", "") or "")
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) > 0:
            ars.append(int(tok))
    primary = int(row.get("NOAA_AR", 0) or 0)
    if primary > 0 and primary not in ars:
        ars.append(primary)
    return ars


# ==============================================================================
# 2. Windowing + pelabelan
# ==============================================================================

def build_labeled_dataset(df_sharp_qc: pd.DataFrame, flares: FlareIndex,
                          cfg: Config) -> pd.DataFrame:
    """Untuk tiap baris SHARP pada waktu t, hitung:

      - `flux_obs`  : fluks maksimum pada [t - obs_window, t)  -> baseline persistence
      - `flux_true` : fluks maksimum pada (t + latency, t + latency + horizon]  -> TARGET

    Window prediksi TIDAK boleh tumpang tindih dengan window observasi, kalau
    tidak model "memprediksi" flare yang sudah terjadi.
    """
    obs_delta = timedelta(hours=cfg.obs_window_hours)
    lat_delta = timedelta(hours=cfg.latency_hours)
    fcst_delta = timedelta(hours=cfg.forecast_horizon_hours)

    ar_lists = [_ar_list(r) for r in df_sharp_qc.to_dict("records")]
    times = df_sharp_qc["T_REC"].tolist()

    flux_obs = np.empty(len(df_sharp_qc))
    flux_true = np.empty(len(df_sharp_qc))
    for i, (ars, t) in enumerate(zip(ar_lists, times)):
        flux_obs[i] = flares.max_flux_multi(ars, t - obs_delta, t)
        flux_true[i] = flares.max_flux_multi(ars, t + lat_delta, t + lat_delta + fcst_delta)

    df = df_sharp_qc.copy()
    df["flux_obs"] = flux_obs
    df["flux_true"] = flux_true
    df["class_obs"] = [flux_to_multiclass_label(f) for f in flux_obs]
    df["class_true"] = [flux_to_multiclass_label(f) for f in flux_true]
    df["y"] = flux_to_binary_label(flux_true, cfg.threshold_class)
    df["y_persistence"] = flux_to_binary_label(flux_obs, cfg.threshold_class)

    pos = int(df["y"].sum())
    rate = pos / max(len(df), 1)
    print(f"[Label] {len(df):,} sampel | positif (>= {cfg.threshold_class}1.0 "
          f"dalam {cfg.forecast_horizon_hours}h): {pos:,} ({rate:.2%})")
    if pos == 0:
        raise RuntimeError(
            "Tidak ada sampel positif. Kemungkinan penyebab: rentang waktu "
            "terlalu pendek/tenang, atau pencocokan NOAA AR gagal."
        )
    return df


# ==============================================================================
# 3. Rekayasa fitur
# ==============================================================================

def add_temporal_features(df: pd.DataFrame, cfg: Config,
                          lookback_hours: int = 24) -> tuple[pd.DataFrame, list[str]]:
    """Tambah fitur laju perubahan per HARP.

    Evolusi (apakah AR sedang tumbuh) membawa informasi yang tidak ada pada
    snapshot tunggal. Semua jendela rolling memakai data MASA LALU saja
    (`closed='left'` tidak dipakai karena nilai saat ini memang teramati pada t).
    """
    df = df.sort_values(["HARPNUM", "T_REC"]).copy()
    base = [c for c in SHARP_FEATURES if c in df.columns]
    steps = max(1, int(round(lookback_hours / cfg.sampling_cadence_hours)))

    new_cols: list[str] = []
    g = df.groupby("HARPNUM", sort=False)
    for col in base:
        d_col = f"{col}_d{lookback_hours}h"
        s_col = f"{col}_std{lookback_hours}h"
        df[d_col] = g[col].diff(steps)
        df[s_col] = g[col].transform(
            lambda s: s.rolling(steps, min_periods=2).std()
        )
        new_cols += [d_col, s_col]

    # Riwayat flare AR itu sendiri adalah prediktor kuat dan gratis.
    df["log_flux_obs"] = np.log10(df["flux_obs"].clip(lower=1e-9))
    new_cols.append("log_flux_obs")

    df[new_cols] = df[new_cols].fillna(0.0)
    return df.reset_index(drop=True), new_cols


def feature_columns(df: pd.DataFrame, include_temporal: bool = False,
                    include_history: bool = False) -> list[str]:
    """Daftar kolom fitur sesuai tingkat kerumitan yang diminta."""
    cols = [c for c in SHARP_FEATURES if c in df.columns and not df[c].isna().all()]
    if include_temporal:
        cols += [c for c in df.columns
                 if c.endswith(("h",)) and ("_d" in c or "_std" in c)
                 and c.split("_d")[0].split("_std")[0] in SHARP_FEATURES]
    if include_history and "log_flux_obs" in df.columns:
        cols.append("log_flux_obs")
    # Buang duplikat sambil mempertahankan urutan
    return list(dict.fromkeys(cols))


# ==============================================================================
# 4. Pembagian data kronologis
# ==============================================================================

def chronological_split(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Bagi train/val/test menurut WAKTU, bukan acak.

    Split acak pada data ini melebih-lebihkan skor secara dramatis: sampel dari
    AR yang sama pada jam-jam berdekatan nyaris identik, sehingga potongan
    "test" acak sebenarnya sudah pernah dilihat model saat latih.
    """
    t_train = pd.Timestamp(cfg.train_end)
    t_val = pd.Timestamp(cfg.val_end)

    train = df[df["T_REC"] <= t_train]
    val = df[(df["T_REC"] > t_train) & (df["T_REC"] <= t_val)]
    test = df[df["T_REC"] > t_val]

    splits = {"train": train, "val": val, "test": test}
    print("\n[Split] Pembagian kronologis:")
    for name, part in splits.items():
        if len(part) == 0:
            print(f"      {name:<6} KOSONG — sesuaikan train_end / val_end di Config")
            continue
        print(f"      {name:<6} {len(part):>8,} sampel | "
              f"{part['T_REC'].min():%Y-%m-%d} .. {part['T_REC'].max():%Y-%m-%d} | "
              f"positif {part['y'].mean():.2%} | {part['HARPNUM'].nunique()} HARP")

    overlap = set(train["HARPNUM"]) & set(test["HARPNUM"])
    if overlap:
        print(f"      catatan: {len(overlap)} HARP muncul di train dan test "
              f"(AR berumur panjang melintasi batas split).")
    print()
    return splits
