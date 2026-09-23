"""Driver pipeline prediksi flare matahari berbasis magnetogram SDO/HMI.

Contoh pemakaian
----------------
    # uji cepat (1 tahun, cadence 6 jam) — sekitar 3-6 menit termasuk unduhan
    python run_pipeline.py --quick

    # jalankan penuh 2010-2018 pada cadence 1 jam (unduhan bisa 1-3 jam)
    python run_pipeline.py

    # pakai cache yang sudah ada, jangan menyentuh jaringan
    python run_pipeline.py --offline

    # tambahkan fitur evolusi 24 jam + riwayat flare AR
    python run_pipeline.py --temporal --history

    # tingkat 4: CNN di atas piksel magnetogram (butuh PyTorch + citra .fits)
    python run_pipeline.py --offline --cnn --cnn-download 4000
    python run_pipeline.py --offline --cnn --cnn-scalars   # CNN + keyword SHARP
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from flare_pipeline import evaluate as ev
from flare_pipeline.acquisition import (fetch_goes_flare_catalog, fetch_sharp_metadata,
                                        fetch_ssw_flares, fill_missing_ar)
from flare_pipeline.config import Config, quick_config
from flare_pipeline.dataset import (
    add_temporal_features,
    build_labeled_dataset,
    chronological_split,
    feature_columns,
    quality_control,
)
from flare_pipeline.labeling import FlareIndex
from flare_pipeline.models import (
    MODEL_FACTORY,
    ClimatologyBaseline,
    PersistenceBaseline,
    ProbabilityCalibrator,
    SingleFeatureThreshold,
    extract_importance,
    unwrap,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def banner(text: str) -> None:
    print(f"\n{'#' * 72}\n#  {text}\n{'#' * 72}")


# ==============================================================================
# Tahap 1-3: data siap latih
# ==============================================================================

def build_dataset(cfg: Config, offline: bool, temporal: bool,
                  history: bool) -> tuple[pd.DataFrame, list[str]]:
    cfg.ensure_dirs()
    cached = cfg.labeled_dataset_path()

    if offline and not cached.exists():
        # `--offline` adalah janji untuk tidak menyentuh jaringan. Kalau cache
        # yang cocok tidak ada, gagal terang-terangan; jangan diam-diam beralih
        # ke mode unduh dan membuat pengguna menunggu tanpa tahu sebabnya.
        available = sorted(p.name for p in cfg.processed_dir.glob("labeled_*.parquet"))
        raise SystemExit(
            f"\n[--offline] Cache yang cocok tidak ditemukan:\n  {cached}\n\n"
            f"Konfigurasi saat ini: {cfg.dataset_signature()}\n\n"
            + (f"Cache yang tersedia:\n" + "".join(f"  {n}\n" for n in available)
               if available else "Belum ada cache dataset berlabel sama sekali.\n")
            + "\nJalankan sekali TANPA --offline untuk membangunnya "
              "(unduhan mentah per bulan tetap dipakai ulang bila sudah ada)."
        )

    if offline:
        df = pd.read_parquet(cached)
        print(f"[Cache] Dataset berlabel dimuat dari {cached.name} ({len(df):,} baris).")
        print("[Cache] Mode offline: tidak ada koneksi jaringan yang dipakai.")
    else:
        banner("TAHAP 1 — AKUISISI DATA")
        df_sharp = fetch_sharp_metadata(cfg)
        df_flares = fetch_goes_flare_catalog(cfg)
        df_flares = fill_missing_ar(df_flares, fetch_ssw_flares(cfg), df_sharp)

        banner("TAHAP 2 — QUALITY CONTROL")
        df_qc = quality_control(df_sharp, cfg)

        banner("TAHAP 3 — WINDOWING & PELABELAN")
        df = build_labeled_dataset(df_qc, FlareIndex(df_flares), cfg)
        df.to_parquet(cached, index=False)
        print(f"[Cache] Dataset berlabel disimpan ke {cached.name}")
        print(f"[Cache] Jalankan lagi dengan --offline untuk melewati "
              f"seluruh Tahap 1-3.")

    if temporal:
        banner("TAHAP 3b — FITUR EVOLUSI TEMPORAL")
        df, _ = add_temporal_features(df, cfg)

    feats = feature_columns(df, include_temporal=temporal, include_history=history)
    print(f"\n[Fitur] {len(feats)} kolom fitur dipakai.")
    return df, feats


# ==============================================================================
# Tahap 4: latih tangga model
# ==============================================================================

def run_models(splits: dict[str, pd.DataFrame], feats: list[str],
               cfg: Config, calibrate: bool = True,
               cnn_opts: dict | None = None) -> tuple[list[dict], dict]:
    train, val, test = splits["train"], splits["val"], splits["test"]
    if len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            "Set validasi atau test kosong. Sesuaikan `train_end` / `val_end` "
            "agar berada di dalam rentang data."
        )

    # Validasi dibelah dua secara kronologis: paruh pertama untuk KALIBRASI,
    # paruh kedua untuk memilih AMBANG. Memakai potongan yang sama untuk
    # keduanya membuat ambang tampak lebih baik daripada kenyataannya.
    val = val.sort_values("T_REC")
    cut = len(val) // 2
    val_cal, val_thr = (val.iloc[:cut], val.iloc[cut:]) if calibrate else (val, val)

    Xtr, ytr = train[feats], train["y"].values
    Xte, yte = test[feats], test["y"].values
    groups_te = test["HARPNUM"].values
    clim = float(ytr.mean())  # basis rate acuan diambil dari set LATIH
    print(f"[Kalibrasi] {'aktif' if calibrate else 'nonaktif'} | "
          f"val kalibrasi {len(val_cal):,} / val ambang {len(val_thr):,}")

    reports: list[dict] = []
    probs_test: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}

    banner("TAHAP 4 — TANGGA MODEL")

    # ---- Tingkat 0a: Climatology ------------------------------------------
    clim_model = ClimatologyBaseline().fit(Xtr, ytr)
    p_te = clim_model.predict_proba(Xte)[:, 1]
    # Ambangnya tidak bermakna (probabilitas konstan); pakai 0.5 apa adanya.
    rep = ev.evaluate(yte, p_te, 0.5, groups_te, clim_model.name, clim)
    reports.append(rep)
    probs_test[clim_model.name] = p_te
    ev.print_report(rep)

    # ---- Tingkat 0b: Persistence ------------------------------------------
    pers = PersistenceBaseline()
    p_te = pers.predict_proba(test["y_persistence"].values)[:, 1]
    rep = ev.evaluate(yte, p_te, 0.5, groups_te, pers.name, clim)
    reports.append(rep)
    probs_test[pers.name] = p_te
    ev.print_report(rep)

    def finish(model, name: str, note: str = "", cols: list[str] | None = None) -> None:
        """Kalibrasi -> pilih ambang di validasi -> nilai sekali di test.

        `cols` memungkinkan satu model memakai kumpulan kolom yang berbeda
        (CNN butuh HARPNUM & T_REC sebagai kunci citra, bukan kolom fitur
        skalar) tanpa mengubah alur evaluasi sedikit pun.
        """
        cols = feats if cols is None else cols
        if calibrate:
            model = ProbabilityCalibrator(model).fit(val_cal[cols], val_cal["y"].values)
        p_thr = model.predict_proba(val_thr[cols])[:, 1]
        thr, tss_va = ev.best_threshold(val_thr["y"].values, p_thr, "TSS")
        print(f"\n[{name}] {note}ambang optimal (validasi) = {thr:.3f}  "
              f"-> TSS validasi {tss_va:+.4f}")

        p_te = model.predict_proba(test[cols])[:, 1]
        rep = ev.evaluate(yte, p_te, thr, groups_te, name, clim)
        reports.append(rep)
        probs_test[name] = p_te
        fitted[name] = model
        ev.print_report(rep)

    # ---- Tingkat 1: ambang satu fitur -------------------------------------
    sft = SingleFeatureThreshold().fit(Xtr, ytr)
    finish(sft, sft.name, note=f"aturan: {sft.describe()}\n{' ' * 4}")

    # ---- Tingkat 2 & 3: model scikit-learn --------------------------------
    for name, factory in MODEL_FACTORY.items():
        model = factory(cfg.random_state)
        model.fit(Xtr, ytr)
        finish(model, name)

    # ---- Tingkat 4: CNN di atas piksel magnetogram ------------------------
    if cnn_opts:
        from flare_pipeline.cnn import CNNClassifier, cnn_columns

        skalar = feats if cnn_opts.get("scalars") else []
        cols = cnn_columns(feats, skalar)
        cnn = CNNClassifier(
            cache=cnn_opts["cache"], scalar_features=skalar,
            epochs=cnn_opts.get("epochs", 40),
            batch_size=cnn_opts.get("batch_size", 64),
            width=cnn_opts.get("width", 16),
            augment=not cnn_opts.get("no_augment", False),
            random_state=cfg.random_state,
        )
        cnn.fit(train[cols], ytr)
        finish(cnn, cnn.name, note=f"{cnn.describe()}\n{' ' * 4}", cols=cols)

    return reports, {"probs": probs_test, "models": fitted,
                     "y_test": yte, "X_test": Xte, "features": feats}


# ==============================================================================
# Tahap 5: laporan & grafik
# ==============================================================================

def make_outputs(reports: list[dict], bundle: dict, cfg: Config,
                 tag: str = "base") -> pd.DataFrame:
    banner("TAHAP 5 — LAPORAN & GRAFIK")

    table = ev.report_table(reports)
    ev.save_reports(reports, cfg.output_dir, f"metrics_test_{tag}")

    show = ["model", "TSS", "HSS", "GSS", "POD", "FAR", "F1",
            "ROC_AUC", "PR_AUC", "BSS", "Accuracy"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("\nRINGKASAN SET TEST (diurutkan menurut TSS):\n")
        print(table[show].sort_values("TSS", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    # Setiap konfigurasi fitur menulis ke subfoldernya sendiri, supaya
    # menjalankan `--temporal` tidak menimpa hasil baseline.
    fig = cfg.figures_dir / tag
    fig.mkdir(parents=True, exist_ok=True)
    yte, probs = bundle["y_test"], bundle["probs"]

    # Baseline climatology & persistence tidak punya kurva probabilistik bermakna
    curve_models = {k: (yte, v) for k, v in probs.items()
                    if not k.startswith(("0a", "0b"))}
    if curve_models:
        ev.plot_roc(curve_models, fig / "roc_curves.png")
        ev.plot_pr(curve_models, fig / "pr_curves.png")

    best = table.sort_values("TSS", ascending=False).iloc[0]
    best_name = best["model"]
    if best_name in probs:
        ev.plot_reliability(yte, probs[best_name], fig / "reliability.png",
                            name=best_name)
        ev.plot_threshold_sweep(yte, probs[best_name], fig / "threshold_sweep.png")
    ev.plot_confusion(best.to_dict(), fig / "confusion_best.png")
    ev.plot_model_comparison(table, fig / "model_comparison_tss.png", "TSS")

    for name, model in bundle["models"].items():
        base = unwrap(model)
        if type(base).__name__ == "CNNClassifier":
            # CNN tidak punya koefisien per fitur; yang informatif justru kurva
            # latihnya — di situ overfitting terlihat sebagai loss yang terus
            # turun sementara skor pantau mandek atau memburuk.
            from flare_pipeline.cnn import plot_training_curve
            plot_training_curve(base, fig / "cnn_training_curve.png")
            continue
        if isinstance(base, SingleFeatureThreshold):
            continue
        vals, label = extract_importance(model, bundle["features"])
        if np.any(vals):
            safe = name.split(". ", 1)[-1].replace(" ", "_").lower()
            ev.plot_feature_importance(bundle["features"], vals,
                                       fig / f"importance_{safe}.png",
                                       title=f"{label} — {name}")

    print(f"\n[Output] Metrik  -> {cfg.output_dir / f'metrics_test_{tag}.csv'}")
    print(f"[Output] Grafik  -> {fig}")
    return table


# ==============================================================================
# Tahap 3c: menyiapkan citra untuk tingkat 4
# ==============================================================================

def prepare_cnn(df: pd.DataFrame, splits: dict[str, pd.DataFrame],
                cfg: Config, args) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Pilih subset citra -> unduh yang kurang -> bangun cache piksel -> samakan split."""
    banner("TAHAP 3c — MENYIAPKAN CITRA UNTUK TINGKAT 4")
    from flare_pipeline import cnn as C
    from flare_pipeline.magnetogram import download_patches

    try:
        h, w = (int(v) for v in args.cnn_size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--cnn-size harus berformat HxW, mis. 128x256 "
                         f"(diterima: {args.cnn_size!r})")
    size = (h, w)

    if args.cnn_ar_blocks:
        subset = C.select_ar_blocks(df, cfg, n_ars=args.cnn_ar_blocks,
                                    max_frames_per_ar=args.cnn_frames_per_ar)
    else:
        subset = C.select_patch_subset(df, cfg, neg_per_pos=args.cnn_neg_per_pos,
                                       max_rows=args.cnn_max_patches)

    if args.cnn_download:
        if args.offline:
            # `--offline` mencakup dataset tabular (tahap 1-3), bukan citra.
            # Unduhan citra adalah permintaan eksplisit tersendiri, jadi
            # dijalankan — tapi jangan sampai pengguna mengira tidak ada
            # jaringan yang tersentuh sama sekali.
            print("[CNN] CATATAN: --offline berlaku untuk dataset tabular. "
                  "--cnn-download yang Anda minta TETAP menghubungi JSOC.")
        kurang = C.missing_patches(subset, cfg)
        print(f"[CNN] {len(kurang):,} citra dari subset belum ada di disk.")
        if len(kurang):
            ambil = kurang.head(args.cnn_download)
            print(f"[CNN] Mengunduh {len(ambil):,} di antaranya "
                  f"(~{len(ambil) * 0.6:.0f} MB). Sisanya bisa diambil dengan "
                  f"menjalankan ulang perintah yang sama.")
            download_patches(ambil, cfg)

    # Cache dibangun dari SELURUH dataset, bukan hanya `subset`. Spesifikasi
    # rentang waktu ikut mengunduh frame "jembatan" yang tidak diminta secara
    # eksplisit; membangun cache dari df membuat frame gratisan itu terpakai
    # alih-alih menganggur di disk. Baris yang berkasnya tidak ada dilewati.
    cache = C.build_patch_cache(df, cfg, size, rebuild=args.cnn_rebuild_cache)
    if len(cache.index) == 0:
        raise SystemExit(
            "\nCache piksel kosong — belum ada satu pun .fits di "
            f"{cfg.fits_dir}.\n\n"
            "Unduh dulu, misalnya:\n"
            "    python run_pipeline.py --cnn --cnn-download 2000\n\n"
            "Tingkat 0-3 tetap bisa dijalankan tanpa citra (tanpa --cnn)."
        )

    if args.cnn_no_restrict:
        print("\n[CNN] PERINGATAN: --cnn-no-restrict aktif. Tingkat 0-3 dinilai "
              "pada seluruh baris, tingkat 4 hanya pada baris bercitra.\n"
              "      Angka TSS antar-tingkat TIDAK bisa dibandingkan langsung.\n")
    else:
        splits = C.restrict_to_cache(splits, cache)

    for nama in ("train", "val", "test"):
        part = splits[nama]
        if len(part) == 0 or part["y"].sum() == 0:
            raise SystemExit(
                f"\nSet {nama} tidak punya sampel positif setelah dibatasi ke "
                f"baris bercitra ({len(part):,} baris).\n"
                "Perbesar cakupan unduhan (--cnn-download lebih besar, atau "
                "--cnn-neg-per-pos lebih kecil agar kuota unduhan terpakai untuk "
                "positif) sebelum melanjutkan ke tingkat 4."
            )

    n_pos_train = int(splits["train"]["y"].sum())
    if n_pos_train < 200:
        print(f"[CNN] CATATAN: hanya {n_pos_train:,} sampel positif di set latih. "
              f"Perlakukan hasil tingkat 4 sebagai uji jalannya pipeline, "
              f"bukan sebagai temuan ilmiah.")

    opts = {"cache": cache, "epochs": args.cnn_epochs, "batch_size": args.cnn_batch,
            "width": args.cnn_width, "scalars": args.cnn_scalars,
            "no_augment": args.cnn_no_augment}
    return opts, splits


# ==============================================================================
# Main
# ==============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="preset 1 tahun / cadence 6 jam untuk uji coba cepat")
    ap.add_argument("--offline", action="store_true",
                    help="pakai dataset berlabel yang sudah di-cache, tanpa jaringan")
    ap.add_argument("--temporal", action="store_true",
                    help="tambahkan fitur perubahan & variabilitas 24 jam")
    ap.add_argument("--history", action="store_true",
                    help="tambahkan riwayat flare AR (log fluks window observasi) sebagai fitur")
    ap.add_argument("--tag", default=None,
                    help="nama untuk berkas keluaran (default: dari kombinasi fitur)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="lewati kalibrasi isotonik (Brier/BSS/reliability jadi tak bermakna)")
    ap.add_argument("--email", default=None, help="email terdaftar JSOC")
    ap.add_argument("--start", default=None, help="waktu mulai, mis. 2012-01-01")
    ap.add_argument("--end", default=None, help="waktu akhir, mis. 2016-12-31")
    ap.add_argument("--horizon", type=int, default=None,
                    help="horizon prakiraan dalam jam (default 24)")
    ap.add_argument("--threshold-class", default=None, choices=["C", "M", "X"],
                    help="ambang kelas flare untuk label positif (default M)")

    g = ap.add_argument_group("tingkat 4 — CNN magnetogram (butuh PyTorch)")
    g.add_argument("--cnn", action="store_true",
                   help="tambahkan CNN di atas piksel magnetogram ke tangga model")
    g.add_argument("--cnn-download", type=int, nargs="?", const=2000, default=0,
                   metavar="N", help="unduh sampai N cutout .fits yang belum ada "
                                     "(butuh jaringan; ~0,6 MB per berkas)")
    g.add_argument("--cnn-neg-per-pos", type=int, default=3,
                   help="rasio negatif:positif saat memilih citra yang diunduh (default 3)")
    g.add_argument("--cnn-ar-blocks", type=int, default=None, metavar="N",
                   help="pilih N wilayah aktif utuh dengan frame KONTIGU, alih-alih "
                        "frame tersebar. Jauh lebih cepat diunduh (satu permintaan "
                        "JSOC per AR, bukan per frame) dan memperlihatkan evolusi AR")
    g.add_argument("--cnn-frames-per-ar", type=int, default=24, metavar="K",
                   help="frame maksimum per AR untuk --cnn-ar-blocks (default 24)")
    g.add_argument("--cnn-max-patches", type=int, default=None, metavar="N",
                   help="batasi subset citra ke N baris, tersebar sepanjang rentang "
                        "waktu. Dipakai untuk uji jalan: batasi di sini (bukan hanya "
                        "di --cnn-download) agar train/val/test sama-sama kebagian citra")
    g.add_argument("--cnn-epochs", type=int, default=40)
    g.add_argument("--cnn-batch", type=int, default=64)
    g.add_argument("--cnn-width", type=int, default=16,
                   help="lebar kanal blok pertama; naikkan hanya bila data banyak")
    g.add_argument("--cnn-size", default="128x256", metavar="HxW",
                   help="ukuran citra setelah resize (default 128x256)")
    g.add_argument("--cnn-scalars", action="store_true",
                   help="gabungkan keyword SHARP ke kepala klasifikasi CNN")
    g.add_argument("--cnn-no-augment", action="store_true",
                   help="matikan augmentasi cermin & geser")
    g.add_argument("--cnn-rebuild-cache", action="store_true",
                   help="bangun ulang cache piksel dari nol")
    g.add_argument("--cnn-no-restrict", action="store_true",
                   help="jangan batasi tingkat 0-3 ke baris bercitra "
                        "(angkanya jadi TIDAK sebanding antar-model)")
    args = ap.parse_args(argv)

    cfg = quick_config() if args.quick else Config()
    if args.email:
        cfg.jsoc_email = args.email
    if args.start:
        cfg.start_time = args.start
    if args.end:
        cfg.end_time = args.end
    if args.horizon:
        cfg.forecast_horizon_hours = args.horizon
    if args.threshold_class:
        cfg.threshold_class = args.threshold_class

    print(f"Rentang   : {cfg.start_time} .. {cfg.end_time} @ {cfg.sampling_cadence_hours}h")
    print(f"Target    : flare >= {cfg.threshold_class}1.0 dalam "
          f"{cfg.forecast_horizon_hours} jam ke depan")
    print(f"Split     : train <= {cfg.train_end} < val <= {cfg.val_end} < test")

    df, feats = build_dataset(cfg, args.offline, args.temporal, args.history)
    splits = chronological_split(df, cfg)

    cnn_opts = None
    if args.cnn:
        cnn_opts, splits = prepare_cnn(df, splits, cfg, args)

    reports, bundle = run_models(splits, feats, cfg,
                                 calibrate=not args.no_calibrate,
                                 cnn_opts=cnn_opts)

    tag = args.tag or ("_".join(
        ["temporal"] * args.temporal + ["history"] * args.history
        + ["cnn"] * args.cnn) or "base")
    make_outputs(reports, bundle, cfg, tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
