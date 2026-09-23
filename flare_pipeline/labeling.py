"""Konversi kelas GOES <-> fluks sinar-X, dan aturan pelabelan target."""

from __future__ import annotations

import bisect
import re

import numpy as np
import pandas as pd

# Kelas GOES adalah skala logaritmik fluks 1-8 Angstrom dalam W/m^2.
# 'M1.0' == 1.0e-5 W/m^2, 'X2.5' == 2.5e-4 W/m^2, dst.
CLASS_EXPONENT = {"A": -8, "B": -7, "C": -6, "M": -5, "X": -4}
_GOES_RE = re.compile(r"^\s*([ABCMX])\s*([0-9]*\.?[0-9]*)\s*$", re.IGNORECASE)


def goes_class_to_flux(cls_str) -> float:
    """'M1.2' -> 1.2e-5. Nilai tidak valid / kosong -> 0.0.

    Konversi ke fluks WAJIB dilakukan sebelum operasi max(). Membandingkan
    string kelasnya secara langsung gagal pada magnitudo dua digit:
    'X10.0' < 'X9.3' menurut urutan teks (karena '1' < '9'), padahal X10.0
    justru lebih dari sepuluh kali lebih kuat. Flare sebesar itu nyata --
    X17 (2003) dan X28 (2003) termasuk yang terkuat yang pernah tercatat.
    """
    if not isinstance(cls_str, str):
        return 0.0
    m = _GOES_RE.match(cls_str)
    if not m:
        return 0.0
    letter = m.group(1).upper()
    mag_str = m.group(2)
    try:
        magnitude = float(mag_str) if mag_str not in ("", ".") else 1.0
    except ValueError:
        magnitude = 1.0
    if magnitude <= 0:
        magnitude = 1.0
    return magnitude * (10.0 ** CLASS_EXPONENT[letter])


def class_threshold_flux(threshold_class: str = "M") -> float:
    """Ambang fluks untuk 'kelas X1.0' -> mis. 'M' -> 1e-5."""
    return 1.0 * (10.0 ** CLASS_EXPONENT[threshold_class.strip().upper()])


def flux_to_multiclass_label(flux: float) -> str:
    """Label kelas terbesar yang tercapai dalam suatu window."""
    if flux >= 1e-4:
        return "X"
    if flux >= 1e-5:
        return "M"
    if flux >= 1e-6:
        return "C"
    if flux >= 1e-7:
        return "B"
    return "Q"  # quiet / tidak ada flare terdeteksi


def flux_to_binary_label(flux, threshold_class: str = "M"):
    """Label biner vektor-aman: 1 jika fluks >= ambang kelas."""
    thr = class_threshold_flux(threshold_class)
    return (np.asarray(flux, dtype=float) >= thr).astype(int)


# ==============================================================================
# Indeks flare per AR untuk lookup window O(log n)
# ==============================================================================

class FlareIndex:
    """Indeks {noaa_ar -> (peak_times terurut, fluks)} agar `max_flux_in_window`
    bisa memakai binary search, bukan scan penuh untuk setiap baris SHARP."""

    def __init__(self, df_flares: pd.DataFrame):
        df = df_flares.copy()
        if "flux" not in df.columns:
            df["flux"] = df["goes_class"].apply(goes_class_to_flux)
        df = df.dropna(subset=["noaa_ar", "peak_time"])
        df = df.sort_values("peak_time")

        self._by_ar: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for ar, g in df.groupby(df["noaa_ar"].astype(int)):
            self._by_ar[int(ar)] = (
                g["peak_time"].to_numpy(dtype="datetime64[ns]"),
                g["flux"].to_numpy(dtype=float),
            )
        # Indeks global (seluruh Matahari) untuk baseline klimatologi harian
        self._all_times = df["peak_time"].to_numpy(dtype="datetime64[ns]")
        self._all_flux = df["flux"].to_numpy(dtype=float)

    def max_flux(self, ar: int, t_start, t_end) -> float:
        """Fluks maksimum untuk AR tertentu pada interval setengah terbuka [t_start, t_end)."""
        entry = self._by_ar.get(int(ar))
        if entry is None:
            return 0.0
        times, fluxes = entry
        lo = np.searchsorted(times, np.datetime64(t_start), side="left")
        hi = np.searchsorted(times, np.datetime64(t_end), side="left")
        if lo >= hi:
            return 0.0
        return float(fluxes[lo:hi].max())

    def max_flux_multi(self, ars, t_start, t_end) -> float:
        """Fluks maksimum di antara beberapa AR (kasus 1 HARP memuat >1 NOAA AR)."""
        best = 0.0
        for ar in ars:
            best = max(best, self.max_flux(ar, t_start, t_end))
        return best

    def known_ars(self) -> set[int]:
        return set(self._by_ar.keys())

    def __len__(self) -> int:
        return len(self._all_times)
