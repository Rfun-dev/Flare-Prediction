"""Akuisisi data: metadata SHARP (JSOC/drms) + katalog flare GOES (HEK/sunpy).

Semua unduhan di-cache per potongan (chunk) bulanan/tahunan di `data/raw/`
sehingga proses bisa dihentikan dan dilanjutkan tanpa mengunduh ulang.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# ==============================================================================
# Format waktu JSOC
# ==============================================================================
# INI SUMBER BUG YANG PALING SERING: JSOC TIDAK menerima ISO-8601.
# Query harus memakai `YYYY.MM.DD_hh:mm:ss_TAI`. Kalau dikirim ISO
# ('2014-01-01T00:00:00Z'), JSOC tidak error — ia hanya mengembalikan
# 0 baris, sehingga kegagalan baru terasa jauh di hilir pipeline.

JSOC_TIME_FMT = "%Y.%m.%d_%H:%M:%S_TAI"


def to_jsoc_time(ts) -> str:
    """Konversi apa pun yang dikenali pandas menjadi timestamp JSOC/TAI."""
    return pd.Timestamp(ts).strftime(JSOC_TIME_FMT)


def parse_t_rec(series: pd.Series) -> pd.Series:
    """Parse kolom T_REC ('2014.01.01_00:00:00_TAI') menjadi datetime64."""
    cleaned = (
        series.astype(str)
        .str.replace("_TAI", "", regex=False)
        .str.replace("_UTC", "", regex=False)
        .str.strip()
    )
    return pd.to_datetime(cleaned, format="%Y.%m.%d_%H:%M:%S", errors="coerce")


# ==============================================================================
# 1. Metadata SHARP dari JSOC
# ==============================================================================

def _month_edges(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Bagi rentang menjadi potongan bulanan (batas kanan eksklusif)."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    edges = pd.date_range(t0.normalize(), t1.normalize() + pd.Timedelta(days=1),
                          freq="MS").tolist()
    if not edges or edges[0] > t0:
        edges.insert(0, t0)
    if edges[-1] < t1:
        edges.append(t1)
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if a < b]


def fetch_sharp_metadata(cfg: Config, force: bool = False,
                         max_retries: int = 3) -> pd.DataFrame:
    """Unduh keyword SHARP untuk SEMUA HARP pada rentang waktu cfg.

    Hanya metadata numerik (bukan citra) — cepat, ringan, dan sudah cukup
    untuk seluruh tangga model klasik. Citra .fits diunduh terpisah lewat
    `magnetogram.download_patches()` bila nanti dipakai CNN.
    """
    import drms

    cfg.ensure_dirs()
    client = drms.Client(email=cfg.jsoc_email)
    key_str = ", ".join(cfg.sharp_keys)
    cad = f"{cfg.sampling_cadence_hours}h"

    frames: list[pd.DataFrame] = []
    for t0, t1 in _month_edges(cfg.start_time, cfg.end_time):
        tag = f"{t0:%Y%m}_{cfg.sharp_series.replace('.', '_')}_{cad}"
        cache = cfg.raw_dir / f"sharp_{tag}.parquet"

        if cache.exists() and not force:
            frames.append(pd.read_parquet(cache))
            print(f"[SHARP] {t0:%Y-%m} cache hit  ({len(frames[-1]):>6} baris)")
            continue

        query = (f"{cfg.sharp_series}[]"
                 f"[{to_jsoc_time(t0)}-{to_jsoc_time(t1)}@{cad}]")

        df = None
        for attempt in range(1, max_retries + 1):
            try:
                df = client.query(query, key=key_str)
                break
            except Exception as exc:  # noqa: BLE001 - JSOC bisa timeout sesaat
                wait = 5 * attempt
                print(f"[SHARP] {t0:%Y-%m} percobaan {attempt} gagal ({exc}); "
                      f"ulangi dalam {wait}s")
                time.sleep(wait)
        if df is None:
            print(f"[SHARP] {t0:%Y-%m} DILEWATI setelah {max_retries} percobaan")
            continue

        if len(df) == 0:
            print(f"[SHARP] {t0:%Y-%m} kosong — periksa query: {query}")
            continue

        df = _coerce_sharp_types(df, cfg)
        df.to_parquet(cache, index=False)
        frames.append(df)
        print(f"[SHARP] {t0:%Y-%m} diunduh  ({len(df):>6} baris) -> {cache.name}")

    if not frames:
        raise RuntimeError(
            "Tidak ada data SHARP sama sekali. Periksa: (1) koneksi internet, "
            "(2) email JSOC sudah terdaftar di "
            "http://jsoc.stanford.edu/ajax/register_email.html, "
            "(3) rentang waktu berada dalam era HMI (>= 2010-05)."
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["HARPNUM", "T_REC"]).reset_index(drop=True)
    print(f"[SHARP] TOTAL {len(out)} baris, {out['HARPNUM'].nunique()} HARP unik.")
    return out


def _coerce_sharp_types(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """JSOC mengembalikan hampir semuanya sebagai string; paksa ke tipe numerik."""
    df = df.copy()
    df["T_REC"] = parse_t_rec(df["T_REC"])
    df = df.dropna(subset=["T_REC"])

    for col in df.columns:
        if col in ("T_REC", "NOAA_ARS"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "NOAA_ARS" in df.columns:
        df["NOAA_ARS"] = df["NOAA_ARS"].astype(str)
    return df.reset_index(drop=True)


# ==============================================================================
# 2. Katalog flare GOES dari HEK
# ==============================================================================

def fetch_goes_flare_catalog(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Unduh daftar kejadian flare GOES beserta asosiasi NOAA AR-nya dari HEK.

    HEK dipilih (bukan berkas GOES XRS mentah) karena setiap kejadian sudah
    dilengkapi `ar_noaanum` — kunci untuk mencocokkan flare ke HARP/SHARP.
    """
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    cfg.ensure_dirs()
    years = range(pd.Timestamp(cfg.start_time).year, pd.Timestamp(cfg.end_time).year + 1)

    frames: list[pd.DataFrame] = []
    for year in years:
        cache = cfg.raw_dir / f"goes_flares_{year}.parquet"
        if cache.exists() and not force:
            frames.append(pd.read_parquet(cache))
            print(f"[HEK] {year} cache hit  ({len(frames[-1]):>5} flare)")
            continue

        t_start = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(cfg.start_time))
        t_end = min(pd.Timestamp(f"{year}-12-31T23:59:59"), pd.Timestamp(cfg.end_time))

        res = Fido.search(
            a.Time(t_start.isoformat(), t_end.isoformat()),
            a.hek.EventType("FL"),
            a.hek.OBS.Observatory == "GOES",
        )
        if len(res) == 0 or len(res["hek"]) == 0:
            print(f"[HEK] {year} tidak ada kejadian.")
            continue

        cols = ["event_starttime", "event_peaktime", "event_endtime",
                "fl_goescls", "ar_noaanum"]
        tbl = res["hek"]
        df = tbl[[c for c in cols if c in tbl.colnames]].to_pandas()
        df = df.rename(columns={
            "event_starttime": "start_time",
            "event_peaktime": "peak_time",
            "event_endtime": "end_time",
            "fl_goescls": "goes_class",
            "ar_noaanum": "noaa_ar",
        })
        df.to_parquet(cache, index=False)
        frames.append(df)
        print(f"[HEK] {year} diunduh  ({len(df):>5} flare) -> {cache.name}")

    if not frames:
        raise RuntimeError("Katalog flare kosong untuk rentang waktu ini.")

    df = pd.concat(frames, ignore_index=True)
    return _clean_flare_catalog(df)


def _clean_flare_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("start_time", "peak_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

    df["goes_class"] = df["goes_class"].astype(str).str.strip()
    df = df[df["goes_class"].str.len() > 0]
    df = df[df["goes_class"].str.upper() != "NAN"]
    df = df.dropna(subset=["peak_time"])

    df["noaa_ar"] = pd.to_numeric(df["noaa_ar"], errors="coerce").fillna(0).astype(int)

    n_before = len(df)
    # HEK memuat duplikat dari beberapa pipeline pelaporan; deduplikasi
    # berdasarkan (waktu puncak dibulatkan ke menit, kelas).
    # Urutkan dulu agar salinan YANG PUNYA nomor AR yang dipertahankan; tanpa
    # ini `drop_duplicates` bisa menyimpan salinan ber-AR 0 dan membuang yang
    # lengkap, sehingga flare itu hilang dari pelabelan tanpa pesan apa pun.
    df["_key"] = df["peak_time"].dt.floor("min").astype(str) + "|" + df["goes_class"]
    df = (df.sort_values("noaa_ar", ascending=False)
            .drop_duplicates(subset="_key").drop(columns="_key"))

    n_unassoc = int((df["noaa_ar"] == 0).sum())
    print(f"[HEK] {n_before} -> {len(df)} flare setelah deduplikasi; "
          f"{n_unassoc} tanpa asosiasi NOAA AR (dipakai hanya untuk klimatologi).")
    return df.sort_values("peak_time").reset_index(drop=True)


# ==============================================================================
# 2b. Menambal nomor AR yang hilang
# ==============================================================================
# Mulai sekitar 2021, laporan flare SWPC yang masuk HEK sebagian besar TIDAK
# lagi menyertakan nomor NOAA AR maupun posisi (tercatat (0,0)). Terukur untuk
# flare M/X: 47% tanpa AR di 2021, 89% di 2022, 85% di 2023 — lalu kembali
# normal (~7%) di 2024. Flare tanpa AR tidak bisa dijodohkan ke HARP, sehingga
# baris AR yang sebenarnya flare diberi label 0. Ini merusak label diam-diam:
# model yang benar menebak flare justru dihitung sebagai false alarm.
#
# Sumber penambalnya "SSW Latest Events" (pipeline SDO/AIA di HEK), yang
# hampir selalu mencantumkan posisi flare dan sering juga nomor AR — walaupun
# nomornya terpotong empat digit (3004 berarti AR 13004).

def _ssw_query(Fido, a, t0, t1, label: str, max_retries: int = 3):
    """Satu kueri HEK untuk FRM 'SSW Latest Events'. None bila gagal total."""
    cols = ["event_peaktime", "fl_goescls", "ar_noaanum", "hgs_x", "hgs_y"]
    for percobaan in range(1, max_retries + 1):
        try:
            res = Fido.search(a.Time(pd.Timestamp(t0).isoformat(),
                                     pd.Timestamp(t1).isoformat()),
                              a.hek.EventType("FL"),
                              a.hek.FRM.Name == "SSW Latest Events")
            if len(res) == 0 or len(res["hek"]) == 0:
                return pd.DataFrame(columns=cols)
            tbl = res["hek"]
            return tbl[[c for c in cols if c in tbl.colnames]].to_pandas()
        except Exception as exc:  # noqa: BLE001 - HEK bisa timeout sesaat
            print(f"[SSW] {label} percobaan {percobaan} gagal ({exc}); ulangi")
            time.sleep(10 * percobaan)
    return None


def fetch_ssw_flares(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Flare 'SSW Latest Events' dari HEK beserta posisi heliografisnya."""
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    cfg.ensure_dirs()
    years = range(pd.Timestamp(cfg.start_time).year, pd.Timestamp(cfg.end_time).year + 1)
    frames: list[pd.DataFrame] = []
    for year in years:
        cache = cfg.raw_dir / f"ssw_flares_{year}.parquet"
        if cache.exists() and not force:
            frames.append(pd.read_parquet(cache))
            print(f"[SSW] {year} cache hit  ({len(frames[-1]):>5} flare)")
            continue

        t_start = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(cfg.start_time))
        t_end = min(pd.Timestamp(f"{year}-12-31T23:59:59"), pd.Timestamp(cfg.end_time))
        df = _ssw_query(Fido, a, t_start, t_end, f"{year}")
        # HEK kadang mengembalikan tabel KOSONG untuk kueri setahun penuh tanpa
        # melempar galat apa pun (terukur pada 2011 dan 2022 — keduanya jelas
        # punya ribuan flare). Kegagalan senyap seperti ini akan tersimpan ke
        # cache sebagai "tahun tanpa flare" dan merusak pelabelan diam-diam,
        # jadi tahun kosong selalu diulang per bulan sebelum dipercaya.
        if df is not None and len(df) == 0:
            print(f"[SSW] {year} kosong pada kueri setahun; mengulang per bulan.")
            potongan = []
            for m0, m1 in _month_edges(t_start, t_end):
                d = _ssw_query(Fido, a, m0, m1, f"{year}-{m0:%m}")
                if d is not None and len(d):
                    potongan.append(d)
            if potongan:
                df = pd.concat(potongan, ignore_index=True)
        if df is None:
            print(f"[SSW] {year} DILEWATI")
            continue
        df = df.rename(columns={"event_peaktime": "peak_time",
                                "fl_goescls": "goes_class", "ar_noaanum": "noaa_ar"})
        df.to_parquet(cache, index=False)
        frames.append(df)
        print(f"[SSW] {year} diunduh  ({len(df):>5} flare) -> {cache.name}")

    if not frames:
        return pd.DataFrame(columns=["peak_time", "goes_class", "noaa_ar", "hgs_x", "hgs_y"])
    df = pd.concat(frames, ignore_index=True)
    df["peak_time"] = pd.to_datetime(df["peak_time"], errors="coerce", utc=True).dt.tz_localize(None)
    df["goes_class"] = df["goes_class"].astype(str).str.strip().str.upper()
    ar = pd.to_numeric(df["noaa_ar"], errors="coerce").fillna(0).astype(int)
    # Nomor AR melewati 10000 pada Juni 2002; seluruh era HMI di atasnya.
    df["noaa_ar"] = np.where((ar > 0) & (ar < 10000), ar + 10000, ar)
    for c in ("hgs_x", "hgs_y"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["peak_time"]).reset_index(drop=True)


def _great_circle_deg(lon1, lat1, lon2, lat2):
    l1, p1, l2, p2 = map(np.radians, (lon1, lat1, lon2, lat2))
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(l1 - l2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _harp_ar_number(noaa_ar, noaa_ars) -> int:
    """Nomor AR utama sebuah baris HARP; jatuh ke NOAA_ARS bila NOAA_AR kosong.

    Terukur: sebagian HARP ber-AR menyimpan nomornya hanya di NOAA_ARS dengan
    NOAA_AR = 0. `quality_control` sudah menerima baris seperti itu, jadi
    penambal juga harus membacanya — kalau tidak, HARP yang benar (sering hanya
    5-7 derajat dari posisi flare) dilewati dan flarenya tetap tanpa AR.
    """
    if noaa_ar and int(noaa_ar) > 0:
        return int(noaa_ar)
    for tok in str(noaa_ars or "").replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) > 0:
            return int(tok)
    return 0


def _nearest_harp_ar(df_sharp: pd.DataFrame, t, lon, lat, max_deg: float,
                     window_hours: float = 3.0) -> tuple[int, float]:
    """NOAA AR dari HARP yang pusat fluksnya paling dekat dengan posisi flare.

    Jendela waktunya +-3 jam, bukan +-1 jam: SHARP punya celah data, dan pada
    +-1 jam sebagian flare tidak menemukan satu baris pun. Rotasi matahari
    dalam 3 jam hanya ~1,6 derajat, jauh di bawah batas jarak yang dipakai.
    """
    cand = df_sharp[(df_sharp["T_REC"] - t).abs() <= pd.Timedelta(hours=window_hours)]
    cand = cand[cand["AR_NUM"] > 0]
    if cand.empty:
        return 0, np.inf
    d = _great_circle_deg(lon, lat, cand["LON_FWT"].to_numpy(), cand["LAT_FWT"].to_numpy())
    i = int(np.nanargmin(d))
    return (int(cand["AR_NUM"].iloc[i]), float(d[i])) if d[i] <= max_deg else (0, float(d[i]))


def fill_missing_ar(df_flares: pd.DataFrame, df_ssw: pd.DataFrame,
                    df_sharp: pd.DataFrame, tol_minutes: float = 10.0,
                    max_deg: float = 10.0, verbose: bool = True) -> pd.DataFrame:
    """Isi `noaa_ar` yang kosong: dari SSW (waktu + kelas sama), lalu dari posisi.

    Dua langkah, dari yang paling bisa dipercaya:

    1. Flare SSW dengan waktu puncak dalam `tol_minutes` dan huruf kelas sama
       yang MENCANTUMKAN nomor AR -> pakai nomor itu.
    2. Kalau SSW hanya memberi posisi, pilih HARP (ber-NOAA) yang pusat fluksnya
       paling dekat pada jam itu, asal jaraknya <= `max_deg` derajat busur.

    `df_sharp` sebaiknya data SHARP SEBELUM filter CMD: flare yang terjadi saat
    AR-nya sudah dekat limb tetap harus bisa melabeli baris AR itu pada jam-jam
    sebelumnya, ketika AR-nya masih lolos filter.
    """
    df = df_flares.copy()
    kosong = np.where(df["noaa_ar"].to_numpy() == 0)[0]
    if len(kosong) == 0 or df_ssw.empty:
        return df

    ssw = df_ssw.sort_values("peak_time").reset_index(drop=True)
    t_ssw = ssw["peak_time"].to_numpy()
    kelas_ssw = ssw["goes_class"].str[:1].to_numpy()
    kol = ["T_REC", "NOAA_AR", "LON_FWT", "LAT_FWT"]
    kol += ["NOAA_ARS"] if "NOAA_ARS" in df_sharp.columns else []
    sharp = df_sharp[kol].dropna(subset=["T_REC", "LON_FWT", "LAT_FWT"])
    nar = pd.to_numeric(sharp["NOAA_AR"], errors="coerce").fillna(0).astype(int)
    nars = sharp["NOAA_ARS"] if "NOAA_ARS" in sharp.columns else pd.Series("", index=sharp.index)
    sharp = sharp.assign(AR_NUM=[_harp_ar_number(a, b) for a, b in zip(nar, nars)])
    sharp = sharp[sharp["AR_NUM"] > 0].sort_values("T_REC").reset_index(drop=True)

    n_ssw_ar = n_posisi = 0
    tol = np.timedelta64(int(tol_minutes * 60), "s")
    for i in kosong:
        t = df.at[i, "peak_time"]
        huruf = str(df.at[i, "goes_class"])[:1].upper()
        lo = np.searchsorted(t_ssw, np.datetime64(t) - tol)
        hi = np.searchsorted(t_ssw, np.datetime64(t) + tol, side="right")
        idx = [j for j in range(lo, hi) if kelas_ssw[j] == huruf]
        if not idx:
            continue
        j = min(idx, key=lambda k: abs(t_ssw[k] - np.datetime64(t)))
        if ssw.at[j, "noaa_ar"] > 0:
            df.at[i, "noaa_ar"] = int(ssw.at[j, "noaa_ar"])
            n_ssw_ar += 1
            continue
        lon, lat = ssw.at[j, "hgs_x"], ssw.at[j, "hgs_y"]
        if pd.isna(lon) or pd.isna(lat) or (lon == 0 and lat == 0) or abs(lon) > 90:
            continue
        ar, _ = _nearest_harp_ar(sharp, t, lon, lat, max_deg)
        if ar:
            df.at[i, "noaa_ar"] = ar
            n_posisi += 1

    if verbose:
        sisa = int((df["noaa_ar"] == 0).sum())
        print(f"[AR-fix] {len(kosong)} flare tanpa AR: {n_ssw_ar} ditambal dari SSW, "
              f"{n_posisi} dari posisi; sisa {sisa} tanpa AR.")
    return df


# ==============================================================================
# 3. Loader dari CSV lokal (jalur alternatif tanpa internet)
# ==============================================================================

def load_sharp_from_csv(path: str | Path, cfg: Config) -> pd.DataFrame:
    """Baca CSV hasil `drms` yang sudah pernah diunduh (mis. `sharp_raw.csv`)."""
    df = pd.read_csv(path)
    return _coerce_sharp_types(df, cfg)
