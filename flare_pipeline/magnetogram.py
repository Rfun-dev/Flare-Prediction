"""Penanganan citra magnetogram (.fits) — jembatan menuju model CNN.

TIDAK dipakai oleh tangga model klasik (tingkat 0-3): seluruh keyword SHARP
sudah merupakan ringkasan terhitung dari magnetogram yang sama, jadi baseline
tidak perlu mengunduh gigabyte citra. Modul ini disiapkan untuk tingkat 4
(CNN / ConvLSTM langsung di atas piksel).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .acquisition import to_jsoc_time
from .config import Config


def patch_filename(harpnum: int, t_rec) -> str:
    """Nama berkas yang dipakai JSOC untuk cutout magnetogram SHARP."""
    ts = pd.Timestamp(t_rec).strftime("%Y%m%d_%H%M%S")
    return f"hmi.sharp_cea_720s.{int(harpnum)}.{ts}_TAI.magnetogram.fits"


def _range_specs(rows: pd.DataFrame, series: str, cadence_hours: int,
                 max_gap_factor: float = 2.0) -> list[str]:
    """Ubah daftar (HARPNUM, T_REC) menjadi spesifikasi record-set JSOC.

    Beda biayanya besar sekali. Tujuh frame berurutan dari satu HARP bisa
    ditulis dua cara:

        [2337][t0]{magnetogram},[2337][t1]{magnetogram},...   ~430 karakter
        [2337][t0-t6@4h]{magnetogram}                          ~80 karakter

    Bentuk kedua bukan cuma lebih pendek — ia satu record-set utuh, yang memang
    dirancang JSOC untuk ekspor massal. Ini penting karena batasan sebenarnya
    di JSOC bukan ukuran unduhan melainkan JUMLAH PERMINTAAN: hanya satu
    permintaan boleh tertunda per akun pada satu waktu, dan tiap permintaan
    butuh beberapa menit. Menekan jumlah permintaan adalah satu-satunya cara
    membuat unduhan berskala besar selesai dalam waktu masuk akal.

    `max_gap_factor` mengendalikan seberapa besar lubang yang boleh dijembatani
    dalam satu rentang. Menjembatani lubang berarti ikut mengunduh frame yang
    tidak diminta; menolak menjembataninya berarti lebih banyak permintaan.
    """
    specs = []
    for h, g in rows.groupby("HARPNUM"):
        t = np.sort(pd.to_datetime(g["T_REC"]).to_numpy())
        if len(t) == 0:
            continue
        jarak = np.diff(t) / np.timedelta64(1, "h")
        potong = np.where(jarak > cadence_hours * max_gap_factor)[0] + 1
        for run in np.split(t, potong):
            if len(run) == 1:
                specs.append(f"{series}[{int(h)}][{to_jsoc_time(run[0])}]{{magnetogram}}")
            else:
                specs.append(f"{series}[{int(h)}]"
                             f"[{to_jsoc_time(run[0])}-{to_jsoc_time(run[-1])}"
                             f"@{cadence_hours}h]{{magnetogram}}")
    return specs


def _batch_by_spec_length(recs: list[str], max_spec_chars: int,
                          max_records: int) -> list[list[str]]:
    """Kelompokkan record-set agar tidak melewati batas panjang JSOC.

    Ada DUA batas berbeda yang gampang tertukar, dan keduanya menolak
    permintaan tanpa mengunduh apa pun:

    1. Panjang URL HTTP. `drms` mengirim record-set sebagai parameter GET, dan
       tiap record membengkak ~2x setelah URL-encoding. Melewatinya menghasilkan
       `HTTP Error 414`.
    2. Panjang spesifikasi record-set di sisi JSOC sendiri, yang JAUH lebih
       ketat daripada batas URL. Melewatinya menghasilkan
       `Record-set specification is too long. [status=4]`.

    Batas kedua yang mengikat, jadi ukuran batch dihitung dari panjang spec
    mentah. Menghitungnya dari jumlah record saja adalah cara paling mudah
    untuk kembali terjebak: panjang tiap record berbeda-beda.
    """
    batches, kini, panjang = [], [], 0
    for r in recs:
        biaya = len(r) + 1                 # +1 untuk koma pemisah
        if kini and (panjang + biaya > max_spec_chars or len(kini) >= max_records):
            batches.append(kini)
            kini, panjang = [], 0
        kini.append(r)
        panjang += biaya
    if kini:
        batches.append(kini)
    return batches


def download_patches(df_sharp: pd.DataFrame, cfg: Config,
                     out_dir: Path | None = None, batch_size: int = 1,
                     skip_existing: bool = True,
                     max_spec_chars: int = 900, max_gap_factor: float = 2.0,
                     max_retries: int = 30, retry_wait: float = 15.0) -> Path:
    """Unduh cutout magnetogram untuk baris SHARP yang diberikan.

    Empat hal yang membuat versi naif gagal atau boros:

      - **Menggabung beberapa record-set dengan koma tidak bisa dipakai untuk
        ekspor.** Terukur: empat record yang masing-masing berhasil diekspor
        sendiri-sendiri akan GAGAL bila dikirim sebagai satu spec berkoma
        (`status=4`, pesan kosong) — jadi ini bukan soal panjang. Karena itu
        `batch_size` default 1, dan penggabungan yang benar-benar bekerja
        adalah spesifikasi RENTANG WAKTU (lihat `_range_specs`): satu
        record-set yang mencakup banyak record sekaligus.
      - Batch yang terlalu panjang juga ditolak — lihat `_batch_by_spec_length`.
        Dipertahankan sebagai pengaman kalau `batch_size` dinaikkan manual.
      - **JSOC hanya mengizinkan SATU permintaan ekspor tertunda per pengguna.**
        Batch berikutnya ditolak dengan `status=7` selama batch sebelumnya
        (atau permintaan dari sesi lain, bahkan dari kemarin) belum tuntas.
        Menganggapnya kegagalan permanen akan membuang seluruh sisa antrean;
        yang benar adalah menunggu lalu mencoba lagi. Perlu diketahui juga:
        permintaan yang GAGAL pun bisa tetap terhitung "tertunda" dan memblokir
        seluruh akun sampai JSOC melepaskannya sendiri — termasuk memblokir
        `method="url_quick"`, jadi tidak ada jalan pintas.
      - `drms` menambah akhiran `.1`, `.2`, ... saat berkas sudah ada, sehingga
        menjalankan ulang skrip menggandakan isi folder. `skip_existing`
        memeriksa keberadaan berkas lebih dulu.
    """
    import drms

    out_dir = Path(out_dir or cfg.fits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = drms.Client(email=cfg.jsoc_email)

    rows = df_sharp[["HARPNUM", "T_REC"]].drop_duplicates()
    if skip_existing:
        exists = rows.apply(
            lambda r: (out_dir / patch_filename(r["HARPNUM"], r["T_REC"])).exists(),
            axis=1)
        n_skip = int(exists.sum())
        rows = rows[~exists]
        if n_skip:
            print(f"[FITS] {n_skip} berkas sudah ada, dilewati.")

    if rows.empty:
        print("[FITS] Tidak ada yang perlu diunduh.")
        return out_dir

    print(f"[FITS] Akan mengunduh {len(rows)} cutout ke {out_dir} "
          f"(~{len(rows) * 0.6:.0f} MB).")

    recs = _range_specs(rows, cfg.sharp_series, cfg.sampling_cadence_hours,
                        max_gap_factor)
    print(f"[FITS] {len(rows):,} frame diringkas menjadi {len(recs):,} record-set.")

    batches = _batch_by_spec_length(recs, max_spec_chars, batch_size)
    print(f"[FITS] {len(batches)} permintaan ekspor. JSOC hanya memproses satu "
          f"permintaan per akun pada satu waktu dan tiap permintaan butuh "
          f"beberapa menit, jadi inilah yang menentukan lamanya.")

    def minta(spec_list: list[str], label: str) -> int:
        """Kirim satu permintaan ekspor. Kembalikan jumlah berkas yang terunduh."""
        spec = ",".join(spec_list)
        for percobaan in range(1, max_retries + 1):
            try:
                req = client.export(spec, method="url", protocol="fits")
                req.wait()
                n = len(req.urls)
                req.download(str(out_dir))
                print(f"[FITS] {label}: {n} berkas selesai.")
                return n
            except Exception as exc:  # noqa: BLE001
                pesan = str(exc)
                if "pending export request" in pesan and percobaan < max_retries:
                    print(f"[FITS] {label}: antrean JSOC masih sibuk, menunggu "
                          f"{retry_wait:.0f}s (percobaan {percobaan}/{max_retries}).")
                    time.sleep(retry_wait)
                    continue
                print(f"[FITS] {label} gagal: {pesan or '(tanpa pesan)'}")
                return 0
        return 0

    selesai = 0
    for i, batch in enumerate(batches, start=1):
        label = f"batch {i}/{len(batches)}"
        n = minta(batch, label)
        if n == 0 and len(batch) > 1:
            # Satu record-set bermasalah menggagalkan SELURUH permintaan, dan
            # JSOC tidak memberi tahu yang mana. Mengulang satu per satu jauh
            # lebih lambat, tapi menyelamatkan record-set yang sebenarnya
            # baik-baik saja — tanpa ini, satu record rusak membuang belasan
            # berkas yang seharusnya bisa diunduh.
            print(f"[FITS] {label}: mengulang {len(batch)} record-set satu per satu.")
            for j, spec in enumerate(batch, start=1):
                n += minta([spec], f"{label}.{j}")
        selesai += n

    print(f"[FITS] Ringkasan: {selesai} berkas terunduh dari {len(rows)} frame diminta.")
    if selesai < len(rows):
        print("[FITS] Jalankan ulang perintah yang sama untuk melanjutkan; "
              "berkas yang sudah ada akan dilewati.")
    return out_dir


def read_patch(path: str | Path) -> np.ndarray:
    """Baca satu magnetogram CEA sebagai array Bz (Gauss)."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return np.array(hdu.data, dtype=np.float32)
    raise ValueError(f"Tidak ada citra 2D di {path}")


def image_features(arr: np.ndarray, strong_field_gauss: float = 100.0) -> dict:
    """Fitur ringkas langsung dari piksel — pemeriksaan silang untuk keyword SHARP
    sekaligus masukan tambahan bila CNN belum dipakai."""
    a = np.nan_to_num(arr, nan=0.0)
    strong = np.abs(a) >= strong_field_gauss
    gy, gx = np.gradient(a)
    grad = np.hypot(gx, gy)

    pos, neg = a[a > 0].sum(), -a[a < 0].sum()
    return {
        "img_total_unsigned_flux": float(np.abs(a).sum()),
        "img_net_flux": float(a.sum()),
        "img_flux_imbalance": float(abs(pos - neg) / max(pos + neg, 1e-9)),
        "img_strong_field_area": float(strong.sum()),
        "img_bz_max": float(np.abs(a).max()),
        "img_bz_std": float(a.std()),
        "img_grad_mean": float(grad.mean()),
        "img_grad_p99": float(np.percentile(grad, 99)),
    }


def resize_patch(arr: np.ndarray, size: tuple[int, int] = (128, 256)) -> np.ndarray:
    """Ubah ukuran ke grid tetap untuk CNN, dengan interpolasi bilinear.

    Bounding box HARP bervariasi ukurannya; CNN butuh tensor seragam. Resize
    (bukan crop) mempertahankan seluruh AR dengan konsekuensi hilangnya skala
    fisik absolut — sertakan AREA_ACR sebagai fitur tambahan bila skala penting.
    """
    from scipy.ndimage import zoom

    a = np.nan_to_num(arr, nan=0.0)
    factors = (size[0] / a.shape[0], size[1] / a.shape[1])
    return zoom(a, factors, order=1).astype(np.float32)


def build_image_feature_table(df_sharp: pd.DataFrame, cfg: Config,
                              fits_dir: Path | None = None) -> pd.DataFrame:
    """Hitung `image_features` untuk setiap baris SHARP yang berkas-nya tersedia."""
    fits_dir = Path(fits_dir or cfg.fits_dir)
    records = []
    for h, t in zip(df_sharp["HARPNUM"], df_sharp["T_REC"]):
        p = fits_dir / patch_filename(h, t)
        if not p.exists():
            continue
        try:
            records.append({"HARPNUM": h, "T_REC": t, **image_features(read_patch(p))})
        except Exception as exc:  # noqa: BLE001
            print(f"[FITS] gagal membaca {p.name}: {exc}")
    print(f"[FITS] {len(records)} citra berhasil diringkas menjadi fitur.")
    return pd.DataFrame.from_records(records)


def dedupe_download_dir(fits_dir: Path | None = None) -> int:
    """Hapus berkas duplikat berakhiran `.1`, `.2`, ... hasil unduhan berulang."""
    fits_dir = Path(fits_dir or Config().fits_dir)
    removed = 0
    for p in fits_dir.glob("*.fits.*"):
        if p.suffix.lstrip(".").isdigit():
            p.unlink()
            removed += 1
    print(f"[FITS] {removed} berkas duplikat dihapus dari {fits_dir}.")
    return removed
