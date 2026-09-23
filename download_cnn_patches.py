"""Unduh cutout magnetogram berskala besar untuk tingkat 4 (CNN).

Kenapa skrip terpisah, bukan sel notebook
-----------------------------------------
Yang menentukan lamanya bukan ukuran berkas melainkan JUMLAH PERMINTAAN EKSPOR.
JSOC hanya memproses satu permintaan tertunda per akun, dan tiap permintaan
butuh beberapa menit. Satu AR dengan frame kontigu = satu permintaan, berapa
pun jumlah framenya. Jadi:

    biaya  ~=  jumlah AR  x  beberapa menit
    hasil  ~=  jumlah AR  x  frame per AR

Menaikkan `--frames` praktis gratis, menaikkan `--n-ars` yang mahal. Tapi
keragaman AR-lah yang menentukan generalisasi, jadi keduanya tetap perlu
dinaikkan bersama-sama — `--frames` hanya memperbaiki rasio hasil per jam.

Skrip ini aman dihentikan kapan saja (Ctrl+C) dan dijalankan ulang: berkas yang
sudah ada dilewati, dan urutan AR-nya deterministik terhadap `random_state`.

Contoh
------
    # lihat rencananya tanpa mengunduh apa pun
    python download_cnn_patches.py --n-ars 300 --frames 40 --dry-run

    # jalankan (berjam-jam; jalankan di latar belakang)
    python download_cnn_patches.py --n-ars 300 --frames 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flare_pipeline.config import Config
from flare_pipeline.magnetogram import download_patches, patch_filename

KOLOM = ["HARPNUM", "T_REC", "y", "NOAA_AR"]


def muat_ringan(cfg: Config, path: Path | None = None) -> pd.DataFrame:
    """Baca dataset berlabel hanya pada kolom yang perlu (hemat memori)."""
    p = path or cfg.labeled_dataset_path()
    if not p.exists():
        kandidat = sorted(cfg.processed_dir.glob("labeled_*.parquet"))
        if not kandidat:
            raise SystemExit("Belum ada dataset berlabel. Jalankan run_pipeline.py dulu.")
        p = kandidat[-1]
        print(f"[Data] {cfg.labeled_dataset_path().name} belum ada; memakai {p.name}")
    df = pd.read_parquet(p, columns=KOLOM)
    print(f"[Data] {len(df):,} baris | {df.HARPNUM.nunique():,} HARP | "
          f"positif {int(df.y.sum()):,} ({df.y.mean():.2%})")
    return df


def pilih_ar(df: pd.DataFrame, n_ars: int, frames: int,
             pos_fraction: float = 0.65, random_state: int = 42) -> pd.DataFrame:
    """Pilih AR lalu ambil potongan waktu KONTIGU di sekitar aktivitas puncaknya.

    Untuk AR berflare, jendelanya dipusatkan pada baris positif pertama supaya
    citra yang diunduh benar-benar memuat kondisi menjelang flare — bukan
    ekor tenang AR yang sama beberapa hari sesudahnya.

    Sebaran tahun ikut diseimbangkan: tanpa itu, pilihan acak akan didominasi
    tahun-tahun ramai (2013-2014 dan 2023-2025) dan model tidak pernah melihat
    AR dari fase tenang.
    """
    rng = np.random.default_rng(random_state)
    df = df.sort_values(["HARPNUM", "T_REC"])
    per_ar = df.groupby("HARPNUM").agg(pernah=("y", "max"),
                                       n=("y", "size"),
                                       t0=("T_REC", "min"))
    per_ar = per_ar[per_ar["n"] >= min(frames, 8)]
    per_ar["tahun"] = per_ar["t0"].dt.year

    n_pos = int(round(n_ars * pos_fraction))
    terpilih: list[int] = []
    for pernah, kuota in ((1, n_pos), (0, n_ars - n_pos)):
        kolam = per_ar[per_ar["pernah"] == pernah]
        if kolam.empty or kuota <= 0:
            continue
        # alokasi merata per tahun, sisa dibagikan ke tahun yang masih punya stok
        tahun = sorted(kolam["tahun"].unique())
        dasar = max(1, kuota // max(len(tahun), 1))
        ambil: list[int] = []
        for th in tahun:
            kandidat = kolam[kolam["tahun"] == th].index.to_numpy()
            k = min(dasar, len(kandidat))
            ambil += list(rng.choice(kandidat, k, replace=False))
        sisa = [h for h in kolam.index if h not in set(ambil)]
        if len(ambil) < kuota and sisa:
            tambah = rng.choice(sisa, min(kuota - len(ambil), len(sisa)),
                                replace=False)
            ambil += list(tambah)
        terpilih += ambil[:kuota]

    potongan = []
    for h in terpilih:
        g = df[df["HARPNUM"] == h]
        if len(g) <= frames:
            potongan.append(g)
            continue
        pos = np.where(g["y"].to_numpy() == 1)[0]
        if len(pos):
            mulai = max(0, int(pos[0]) - frames // 2)
        else:
            mulai = max(0, (len(g) - frames) // 2)
        potongan.append(g.iloc[mulai:mulai + frames])

    out = pd.concat(potongan, ignore_index=True)
    print(f"[Pilih] {out.HARPNUM.nunique()} AR | {len(out):,} frame | "
          f"positif {int(out.y.sum()):,} ({out.y.mean():.1%})")
    print("[Pilih] sebaran tahun:")
    tab = out.groupby(out["T_REC"].dt.year).agg(
        AR=("HARPNUM", "nunique"), frame=("y", "size"), positif=("y", "sum"))
    print(tab.to_string())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-ars", type=int, default=300)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--pos-fraction", type=float, default=0.65)
    ap.add_argument("--dry-run", action="store_true",
                    help="tampilkan rencana lalu berhenti, tanpa mengunduh")
    ap.add_argument("--dataset", default=None,
                    help="path parquet berlabel tertentu (default: sesuai Config)")
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.ensure_dirs()
    df = muat_ringan(cfg, Path(args.dataset) if args.dataset else None)
    subset = pilih_ar(df, args.n_ars, args.frames, args.pos_fraction,
                      cfg.random_state)

    ada = np.fromiter(
        ((cfg.fits_dir / patch_filename(h, t)).exists()
         for h, t in zip(subset.HARPNUM, subset.T_REC)),
        dtype=bool, count=len(subset))
    kurang = subset[~ada]
    print(f"\n[Status] {int(ada.sum()):,} frame sudah ada, "
          f"{len(kurang):,} perlu diunduh "
          f"({kurang.HARPNUM.nunique()} AR, ~{len(kurang) * 0.6 / 1024:.1f} GB).")
    print(f"[Status] Perkiraan waktu: {kurang.HARPNUM.nunique()} permintaan ekspor "
          f"x beberapa menit = berjam-jam. Skrip ini bisa dihentikan dan "
          f"dilanjutkan kapan saja.")

    if args.dry_run or kurang.empty:
        return 0

    t0 = time.time()
    # Unduh per AR supaya kemajuannya terlihat dan berhenti di tengah jalan
    # tidak membuang apa pun. download_patches sendiri sudah melewati berkas
    # yang ada dan menunggu bila antrean JSOC sibuk.
    harps = kurang["HARPNUM"].drop_duplicates().to_numpy()
    for i, h in enumerate(harps, start=1):
        g = kurang[kurang["HARPNUM"] == h]
        lewat = time.time() - t0
        sisa = (lewat / max(i - 1, 1)) * (len(harps) - i + 1) if i > 1 else 0
        print(f"\n[{i}/{len(harps)}] HARP {h} ({len(g)} frame) | "
              f"lewat {lewat / 60:.0f} mnt | perkiraan sisa {sisa / 60:.0f} mnt")
        try:
            download_patches(g, cfg)
        except KeyboardInterrupt:
            print("\nDihentikan. Jalankan ulang untuk melanjutkan.")
            return 130
        except Exception as exc:  # noqa: BLE001
            print(f"[FITS] HARP {h} dilewati karena galat: {exc}")

    print(f"\nSelesai dalam {(time.time() - t0) / 60:.0f} menit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
