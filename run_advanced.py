"""Model lanjutan: mengisolasi kontribusi tiap saran di Bagian 10 notebook.

Pertanyaannya bukan "berapa TSS model lanjutan", melainkan "apakah tambahan
fitur itu benar-benar menolong, dan seberapa yakin kita?". Karena itu skrip ini
melatih beberapa konfigurasi fitur pada baris yang **identik**, lalu menguji
selisihnya dengan bootstrap berpasangan.

Konfigurasi fitur yang dibandingkan
-----------------------------------
    A. SHARP saja          18 kolom   <- sama dengan tingkat 2-3 di notebook
    B. + fitur temporal    54 kolom   <- saran #2 (add_temporal_features)
    C. + riwayat flare     55 kolom   <- saran #3 (include_history=True)

Contoh pemakaian
----------------
    # rentang penuh 2010-2018, pakai cache yang sudah dibangun run_pipeline.py
    python run_advanced.py --offline

    # rentang pendek untuk uji coba cepat
    python run_advanced.py --quick --offline

    # hanya regresi logistik (jauh lebih cepat daripada seluruh ladder)
    python run_advanced.py --offline --models logistic

    # DISARANKAN: evaluasi rolling-origin, beberapa jendela uji berurutan
    python run_advanced.py --offline --walk-forward 5 --test-months 12

Kenapa walk-forward disarankan
------------------------------
Split tunggal bawaan `Config()` menempatkan set uji di 2017-2018, yaitu dasar
minimum matahari. Di jendela itu seluruh kejadian positif berasal dari 3 HARP
saja, sehingga bootstrap per HARP kehabisan blok positif dan selang TSS melebar
sampai ~0,99 — praktis tidak ada kesimpulan yang bisa ditarik. Menggabungkan
tiga jendela uji menyempitkannya ke ~0,12.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings

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
    PersistenceBaseline,
    ProbabilityCalibrator,
    extract_importance,
    unwrap,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


LOGIT_TUNED = "2b. Regresi logistik (C disetel)"


def banner(text: str) -> None:
    print(f"\n{'#' * 72}\n#  {text}\n{'#' * 72}")


# ==============================================================================
# Regresi logistik dengan regularisasi yang disetel
# ==============================================================================

def make_logistic_tuned(random_state: int, X_tr, y_tr, X_sel, y_sel,
                        grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    """Regresi logistik yang kekuatan regularisasinya dipilih di data validasi.

    `make_logistic` di models.py mematok C=1.0. Itu wajar untuk 18 fitur SHARP
    yang saling lepas, tapi konfigurasi B dan C menaikkan jumlah kolom menjadi
    54-55 dan kolom-kolom itu sangat berkorelasi (selisih 24 jam sebuah besaran
    jelas berkaitan dengan besaran itu sendiri). Pada jumlah kolom seperti itu
    C=1.0 praktis tanpa rem, dan model linearnya mengejar derau.

    C dipilih memakai average precision di potongan validasi, BUKAN lewat
    `LogisticRegressionCV`: cross-validation bawaannya membagi data secara acak,
    dan pada data ini pembagian acak bocor karena baris AR yang sama pada jam
    berdekatan hampir identik.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from flare_pipeline.models import SignedLog

    terbaik, skor_terbaik = None, -np.inf
    for C in grid:
        pipe = Pipeline([
            ("log", SignedLog()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=5000,
                                       class_weight="balanced",
                                       solver="lbfgs", random_state=random_state)),
        ]).fit(X_tr, y_tr)
        ap = average_precision_score(y_sel, pipe.predict_proba(X_sel)[:, 1])
        if ap > skor_terbaik:
            terbaik, skor_terbaik, C_terbaik = pipe, ap, C
    print(f"      (C terpilih {C_terbaik:g}, AP validasi {skor_terbaik:.4f})")
    return terbaik


# ==============================================================================
# Uji beda BERPASANGAN
# ==============================================================================

def paired_delta_ci(y_true, pred_a, pred_b, groups, metric: str = "TSS",
                    n_boot: int = 1000, alpha: float = 0.05,
                    random_state: int = 42) -> dict:
    """Selang kepercayaan untuk SELISIH metrik dua model pada baris yang sama.

    Kenapa berpasangan, bukan membandingkan dua selang marginal: kedua model
    dinilai pada baris yang persis sama, sehingga sebagian besar ragam skornya
    berasal dari "kebetulan set uji ini berisi AR yang mana" — dan ragam itu
    dialami KEDUA model bersama-sama. Dengan meresample HARP lalu menghitung
    selisihnya di dalam resample yang sama, komponen ragam bersama itu saling
    meniadakan.

    Akibatnya selang selisih bisa jauh lebih sempit daripada selang masing-masing
    model, dan sebuah perbaikan nyata tetap terdeteksi walaupun dua selang
    marginalnya bertumpang tindih hampir seluruhnya.

    Mengembalikan delta (b - a), selangnya, dan porsi resample yang delta-nya
    positif (dibaca seperti peluang posterior bahwa b lebih baik dari a).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)

    groups = np.asarray(groups)
    idx_pool = [np.where(groups == g)[0] for g in np.unique(groups)]
    n_blocks = len(idx_pool)

    m = lambda yt, yp: ev.deterministic_metrics(yt, yp)[metric]  # noqa: E731
    obs = m(y_true, pred_b) - m(y_true, pred_a)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([idx_pool[p] for p in pick])
        yt = y_true[idx]
        deltas[i] = m(yt, pred_b[idx]) - m(yt, pred_a[idx])

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "delta": float(obs),
        "lo": float(lo),
        "hi": float(hi),
        "p_better": float((deltas > 0).mean()),
        "signifikan": bool(lo > 0 or hi < 0),
    }


# ==============================================================================
# Evaluasi walk-forward (rolling origin)
# ==============================================================================

def walk_forward_splits(df: pd.DataFrame, n_folds: int = 4,
                        val_months: int = 6, test_months: int = 12,
                        min_train_months: int = 36) -> list[dict]:
    """Jendela mengembang: latih [awal, t), validasi [t, t+v), uji [t+v, t+v+u).

    Kenapa ini perlu, bukan sekadar memperpanjang rentang: satu set uji tunggal
    di ujung rentang mempertaruhkan seluruh kesimpulan pada fase siklus yang
    kebetulan ada di situ. Dengan `Config()` apa adanya, set ujinya jatuh di
    2017-2018 (minimum matahari) dan SELURUH kejadian positifnya datang dari
    3 HARP saja — bootstrap per HARP praktis kehabisan blok positif.

    Beberapa jendela uji berurutan mengumpulkan jauh lebih banyak AR berflare
    yang berbeda, dan itulah yang sebenarnya menentukan lebar selang.
    """
    t0 = df["T_REC"].min().normalize()
    t1 = df["T_REC"].max().normalize()

    folds = []
    for k in range(n_folds):
        # Jendela uji dimundurkan satu per satu dari ujung rentang.
        test_end = t1 - pd.DateOffset(months=test_months * (n_folds - 1 - k))
        test_start = test_end - pd.DateOffset(months=test_months)
        val_start = test_start - pd.DateOffset(months=val_months)
        if val_start - t0 < pd.Timedelta(days=30 * min_train_months):
            continue

        tr = df[df["T_REC"] < val_start]
        va = df[(df["T_REC"] >= val_start) & (df["T_REC"] < test_start)]
        te = df[(df["T_REC"] >= test_start) & (df["T_REC"] < test_end)]
        if not len(tr) or not len(va) or not len(te):
            continue
        if va["y"].sum() < 10 or te["y"].sum() < 10:
            print(f"      fold {k}: dilewati (positif terlalu sedikit: "
                  f"val {int(va['y'].sum())}, uji {int(te['y'].sum())})")
            continue

        folds.append({"fold": len(folds), "train": tr, "val": va, "test": te,
                      "rentang_uji": f"{test_start:%Y-%m}..{test_end:%Y-%m}"})
    return folds


def jalankan_walk_forward(konfigurasi: dict, folds: list[dict], cfg: Config,
                          model_keys: list[str]) -> tuple[list[dict], dict]:
    """Latih ulang tiap fold, lalu gabungkan prediksi seluruh jendela uji.

    Metrik dihitung pada gabungan itu. Blok bootstrap diberi nama
    `fold_HARPNUM` supaya satu AR yang muncul di dua jendela uji tidak dianggap
    blok yang sama.
    """
    kumpul: dict[str, dict] = {}
    baris_fold = []

    for f in folds:
        tr, va, te = f["train"], f["val"], f["test"]
        va = va.sort_values("T_REC")
        cut = len(va) // 2
        val_cal, val_thr = va.iloc[:cut], va.iloc[cut:]
        clim = float(tr["y"].mean())
        grup = np.array([f"{f['fold']}_{h}" for h in te["HARPNUM"].values])

        print(f"\n  fold {f['fold']} | uji {f['rentang_uji']} | "
              f"latih {len(tr):,} / val {len(va):,} / uji {len(te):,} | "
              f"positif uji {int(te['y'].sum())} dari "
              f"{te.loc[te['y'] == 1, 'HARPNUM'].nunique()} HARP")

        for nama, feats in konfigurasi.items():
            for key in model_keys:
                if key == LOGIT_TUNED:
                    model = make_logistic_tuned(cfg.random_state, tr[feats],
                                                tr["y"].values, val_cal[feats],
                                                val_cal["y"].values)
                else:
                    model = MODEL_FACTORY[key](cfg.random_state)
                    model.fit(tr[feats], tr["y"].values)
                model = ProbabilityCalibrator(model).fit(val_cal[feats],
                                                         val_cal["y"].values)
                thr, _ = ev.best_threshold(
                    val_thr["y"].values,
                    model.predict_proba(val_thr[feats])[:, 1], "TSS")
                p = model.predict_proba(te[feats])[:, 1]

                label = f"{nama} | {key.split('. ', 1)[-1]}"
                d = kumpul.setdefault(label, {"y": [], "prob": [], "biner": [],
                                              "grup": [], "clim": []})
                d["y"].append(te["y"].values)
                d["prob"].append(p)
                d["biner"].append((p >= thr).astype(int))
                d["grup"].append(grup)
                d["clim"].append(clim)

                r = ev.deterministic_metrics(te["y"].values, (p >= thr).astype(int))
                baris_fold.append({"fold": f["fold"], "rentang_uji": f["rentang_uji"],
                                   "model": label, "TSS": r["TSS"], "HSS": r["HSS"],
                                   "n_uji": len(te), "pos_uji": int(te["y"].sum())})

    # ---- gabungkan seluruh fold ------------------------------------------
    reports, preds = [], {}
    for label, d in kumpul.items():
        y = np.concatenate(d["y"])
        prob = np.concatenate(d["prob"])
        biner = np.concatenate(d["biner"])
        grup = np.concatenate(d["grup"])
        rep = {"model": label, "threshold": np.nan}
        rep.update(ev.deterministic_metrics(y, biner))
        rep.update(ev.probabilistic_metrics(y, prob, float(np.mean(d["clim"]))))
        for m in ("TSS", "HSS"):
            lo, hi = ev.bootstrap_ci(y, biner, grup, metric=m, n_boot=300)
            rep[f"{m}_lo"], rep[f"{m}_hi"] = lo, hi
        rep["konfigurasi"] = label.split(" | ")[0]
        rep["n_fitur"] = len(konfigurasi.get(label.split(" | ")[0], []))
        reports.append(rep)
        preds[label] = {"biner": biner, "prob": prob, "y": y, "grup": grup,
                        "model": None, "feats": []}

    return reports, {"preds": preds, "per_fold": pd.DataFrame(baris_fold)}


# ==============================================================================
# Data
# ==============================================================================

def build_dataset(cfg: Config, offline: bool) -> pd.DataFrame:
    cfg.ensure_dirs()
    cached = cfg.labeled_dataset_path()

    if cached.exists():
        df = pd.read_parquet(cached)
        print(f"[Cache] Dataset berlabel dimuat dari {cached.name} "
              f"({len(df):,} baris).")
        return df

    if offline:
        tersedia = sorted(p.name for p in cfg.processed_dir.glob("labeled_*.parquet"))
        raise SystemExit(
            f"\n[--offline] Cache tidak ditemukan: {cached.name}\n"
            f"Konfigurasi saat ini: {cfg.dataset_signature()}\n\n"
            + ("Cache yang ada:\n" + "".join(f"  {n}\n" for n in tersedia)
               if tersedia else "Belum ada cache sama sekali.\n")
            + "\nBangun dulu dengan: python run_pipeline.py\n"
        )

    banner("TAHAP 1-3 — AKUISISI, QC, PELABELAN")
    df_sharp = fetch_sharp_metadata(cfg)
    df_flares = fetch_goes_flare_catalog(cfg)
    df_flares = fill_missing_ar(df_flares, fetch_ssw_flares(cfg), df_sharp)
    df_qc = quality_control(df_sharp, cfg)
    df = build_labeled_dataset(df_qc, FlareIndex(df_flares), cfg)
    df.to_parquet(cached, index=False)
    print(f"[Cache] Disimpan ke {cached.name}")
    return df


# ==============================================================================
# Pelatihan satu konfigurasi fitur
# ==============================================================================

def jalankan_konfig(nama: str, feats: list[str], splits: dict, cfg: Config,
                    model_keys: list[str]) -> tuple[list[dict], dict]:
    """Latih semua model pada satu kumpulan fitur, nilai di set uji yang sama."""
    train, val, test = splits["train"], splits["val"], splits["test"]

    val = val.sort_values("T_REC")
    cut = len(val) // 2
    val_cal, val_thr = val.iloc[:cut], val.iloc[cut:]

    Xtr, ytr = train[feats], train["y"].values
    yte = test["y"].values
    groups_te = test["HARPNUM"].values
    clim = float(ytr.mean())

    reports, preds = [], {}
    for key in model_keys:
        if key == LOGIT_TUNED:
            model = make_logistic_tuned(cfg.random_state, Xtr, ytr,
                                        val_cal[feats], val_cal["y"].values)
        else:
            model = MODEL_FACTORY[key](cfg.random_state)
            model.fit(Xtr, ytr)
        model = ProbabilityCalibrator(model).fit(val_cal[feats], val_cal["y"].values)

        thr, tss_va = ev.best_threshold(
            val_thr["y"].values, model.predict_proba(val_thr[feats])[:, 1], "TSS")
        p_te = model.predict_proba(test[feats])[:, 1]

        label = f"{nama} | {key.split('. ', 1)[-1]}"
        rep = ev.evaluate(yte, p_te, thr, groups_te, label, clim)
        rep["konfigurasi"] = nama
        rep["n_fitur"] = len(feats)
        reports.append(rep)
        preds[label] = {
            "biner": (p_te >= thr).astype(int),
            "prob": p_te,
            "model": model,
            "feats": feats,
        }
        print(f"  {label:<42} TSS {rep['TSS']:+.4f} "
              f"[{rep['TSS_lo']:+.3f}, {rep['TSS_hi']:+.3f}]   "
              f"HSS {rep['HSS']:+.4f}   (ambang {thr:.3f}, TSS val {tss_va:+.4f})")

    return reports, preds


# ==============================================================================
# Main
# ==============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="rentang pendek 2013-2014 (preset quick_config)")
    ap.add_argument("--offline", action="store_true",
                    help="wajib ada cache; jangan menyentuh jaringan")
    ap.add_argument("--models", default="all",
                    choices=["all", "logistic", "trees", "tuned"],
                    help="model mana yang dilatih per konfigurasi "
                         "('tuned' menambahkan regresi logistik ber-regularisasi)")
    ap.add_argument("--lookback", type=int, default=24,
                    help="jendela fitur temporal dalam jam (default 24)")
    ap.add_argument("--walk-forward", type=int, default=0, metavar="K",
                    help="evaluasi rolling-origin dengan K jendela uji berurutan "
                         "(0 = pakai split tunggal biasa)")
    ap.add_argument("--test-months", type=int, default=12,
                    help="panjang tiap jendela uji walk-forward, dalam bulan")
    ap.add_argument("--min-train-months", type=int, default=36,
                    help="panjang minimum data latih sebelum fold dianggap sah")
    ap.add_argument("--tag", default="lanjutan", help="nama subfolder keluaran")
    ap.add_argument("--email", default=None)
    args = ap.parse_args(argv)

    cfg = quick_config() if args.quick else Config()
    if args.email:
        cfg.jsoc_email = args.email
    cfg.ensure_dirs()

    print(f"Rentang   : {cfg.start_time} .. {cfg.end_time} @ "
          f"{cfg.sampling_cadence_hours}h")
    print(f"Target    : flare >= {cfg.threshold_class}1.0 dalam "
          f"{cfg.forecast_horizon_hours} jam")

    df = build_dataset(cfg, args.offline)

    banner("FITUR EVOLUSI TEMPORAL")
    df, kolom_baru = add_temporal_features(df, cfg, lookback_hours=args.lookback)
    print(f"[Temporal] {len(kolom_baru)} kolom baru "
          f"(jendela {args.lookback} jam, {args.lookback // cfg.sampling_cadence_hours} langkah).")

    konfigurasi = {
        "A. SHARP saja": feature_columns(df),
        "B. + temporal": feature_columns(df, include_temporal=True),
        "C. + riwayat": feature_columns(df, include_temporal=True,
                                        include_history=True),
    }
    for nama, cols in konfigurasi.items():
        print(f"           {nama:<16} {len(cols):>3} fitur")

    model_keys = {
        "all": list(MODEL_FACTORY) + [LOGIT_TUNED],
        "logistic": ["2. Regresi logistik"],
        "trees": ["3a. Random Forest", "3b. Gradient Boosting"],
        "tuned": ["2. Regresi logistik", LOGIT_TUNED],
    }[args.models]

    # ======================================================================
    # Mode walk-forward
    # ======================================================================
    if args.walk_forward:
        banner(f"TAHAP 4 — WALK-FORWARD {args.walk_forward} JENDELA")
        folds = walk_forward_splits(df, n_folds=args.walk_forward,
                                    test_months=args.test_months,
                                    min_train_months=args.min_train_months)
        if not folds:
            raise SystemExit("Tidak ada fold yang memenuhi syarat. "
                             "Kecilkan --test-months atau --walk-forward.")
        print(f"[Walk-forward] {len(folds)} fold terbentuk.")

        semua_report, bundel = jalankan_walk_forward(konfigurasi, folds, cfg,
                                                     model_keys)
        semua_pred = bundel["preds"]
        per_fold = bundel["per_fold"]

        banner("TAHAP 5 — APAKAH TAMBAHAN FITUR BENAR-BENAR MENOLONG?")
        print("\nBootstrap berpasangan pada gabungan seluruh jendela uji.\n")
        deltas = []
        for key in model_keys:
            singkat = key.split(". ", 1)[-1]
            rantai = [f"{n} | {singkat}" for n in konfigurasi]
            for a, b in itertools.combinations(rantai, 2):  # termasuk A->C
                if a not in semua_pred or b not in semua_pred:
                    continue
                for metric in ("TSS", "HSS"):
                    d = paired_delta_ci(semua_pred[a]["y"], semua_pred[a]["biner"],
                                        semua_pred[b]["biner"],
                                        semua_pred[a]["grup"], metric=metric)
                    d.update({"model": singkat, "dari": a.split(" | ")[0],
                              "ke": b.split(" | ")[0]})
                    deltas.append(d)
                    tanda = "NYATA  " if d["signifikan"] else "belum  "
                    print(f"  {singkat:<28} {d['dari']:<14} -> {d['ke']:<14} "
                          f"d{metric} {d['delta']:+.4f} "
                          f"[{d['lo']:+.4f}, {d['hi']:+.4f}]  "
                          f"p(lebih baik) {d['p_better']:.2f}  {tanda}")

        banner("TAHAP 6 — LAPORAN")
        table = ev.report_table(semua_report)
        ev.save_reports(semua_report, cfg.output_dir, f"metrics_test_{args.tag}")
        p_fold = cfg.output_dir / f"per_fold_{args.tag}.csv"
        per_fold.to_csv(p_fold, index=False)
        pd.DataFrame(deltas).to_csv(cfg.output_dir / f"delta_{args.tag}.csv",
                                    index=False)

        show = [c for c in ["model", "n_fitur", "TSS", "TSS_lo", "TSS_hi", "HSS",
                            "POD", "FAR", "ROC_AUC", "PR_AUC", "N", "base_rate"]
                if c in table.columns]
        with pd.option_context("display.width", 220, "display.max_columns", 50):
            print("\nGABUNGAN SELURUH JENDELA UJI (diurutkan menurut TSS):\n")
            print(table[show].sort_values("TSS", ascending=False)
                  .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
            print("\nTSS PER FOLD:\n")
            piv = per_fold.pivot_table(index="model", columns="rentang_uji",
                                       values="TSS")
            print(piv.to_string(float_format=lambda v: f"{v: .3f}"))

        fig = cfg.figures_dir / args.tag
        fig.mkdir(parents=True, exist_ok=True)
        ev.plot_model_comparison(table, fig / "model_comparison_tss.png", "TSS")
        kurva = {k: (v["y"], v["prob"]) for k, v in semua_pred.items()}
        ev.plot_roc(kurva, fig / "roc_curves.png")
        ev.plot_pr(kurva, fig / "pr_curves.png")
        print(f"\n[Grafik] {fig}\n[Metrik] {cfg.output_dir}")
        return 0

    # ======================================================================
    # Mode split tunggal
    # ======================================================================
    splits = chronological_split(df, cfg)
    test = splits["test"]
    yte = test["y"].values
    groups_te = test["HARPNUM"].values

    # ---- pembanding tetap: Persistence ------------------------------------
    banner("TAHAP 4 — TANGGA MODEL PER KONFIGURASI FITUR")
    clim = float(splits["train"]["y"].mean())
    pers = PersistenceBaseline()
    p_pers = pers.predict_proba(test["y_persistence"].values)[:, 1]
    rep_pers = ev.evaluate(yte, p_pers, 0.5, groups_te, pers.name, clim)
    rep_pers["konfigurasi"] = "pembanding"
    rep_pers["n_fitur"] = 0
    print(f"\n  {pers.name:<42} TSS {rep_pers['TSS']:+.4f} "
          f"[{rep_pers['TSS_lo']:+.3f}, {rep_pers['TSS_hi']:+.3f}]   "
          f"HSS {rep_pers['HSS']:+.4f}")

    semua_report = [rep_pers]
    semua_pred = {pers.name: {"biner": (p_pers >= 0.5).astype(int),
                              "prob": p_pers, "model": None, "feats": []}}

    for nama, cols in konfigurasi.items():
        print(f"\n--- {nama} ({len(cols)} fitur) ---")
        reps, preds = jalankan_konfig(nama, cols, splits, cfg, model_keys)
        semua_report += reps
        semua_pred.update(preds)

    # ---- uji beda berpasangan ---------------------------------------------
    banner("TAHAP 5 — APAKAH TAMBAHAN FITUR BENAR-BENAR MENOLONG?")
    print("\nBootstrap berpasangan per HARP, 1000 resample, pada baris uji yang sama.")
    print("Selang yang tidak memuat nol berarti selisihnya nyata.\n")

    deltas = []
    for key in model_keys:
        singkat = key.split(". ", 1)[-1]
        rantai = [f"{n} | {singkat}" for n in konfigurasi]
        for a, b in itertools.combinations(rantai, 2):  # termasuk A->C
            if a not in semua_pred or b not in semua_pred:
                continue
            for metric in ("TSS", "HSS"):
                d = paired_delta_ci(yte, semua_pred[a]["biner"],
                                    semua_pred[b]["biner"], groups_te,
                                    metric=metric)
                d["model"] = singkat
                d["dari"] = a.split(" | ")[0]
                d["ke"] = b.split(" | ")[0]
                deltas.append(d)
                tanda = "NYATA  " if d["signifikan"] else "belum  "
                print(f"  {singkat:<20} {d['dari']:<14} -> {d['ke']:<14} "
                      f"d{metric} {d['delta']:+.4f} "
                      f"[{d['lo']:+.4f}, {d['hi']:+.4f}]  "
                      f"p(lebih baik) {d['p_better']:.2f}  {tanda}")

    # ---- keluaran ----------------------------------------------------------
    banner("TAHAP 6 — LAPORAN & GRAFIK")
    table = ev.report_table(semua_report)
    ev.save_reports(semua_report, cfg.output_dir, f"metrics_test_{args.tag}")

    df_delta = pd.DataFrame(deltas)
    if len(df_delta):
        p_delta = cfg.output_dir / f"delta_{args.tag}.csv"
        df_delta.to_csv(p_delta, index=False)
        print(f"[Simpan] {p_delta.name}")

    show = ["model", "n_fitur", "TSS", "TSS_lo", "TSS_hi", "HSS", "POD", "FAR",
            "ROC_AUC", "PR_AUC", "BSS", "N", "base_rate"]
    show = [c for c in show if c in table.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print("\nRINGKASAN SET TEST (diurutkan menurut TSS):\n")
        print(table[show].sort_values("TSS", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    fig = cfg.figures_dir / args.tag
    fig.mkdir(parents=True, exist_ok=True)

    ev.plot_model_comparison(table, fig / "model_comparison_tss.png", "TSS")
    kurva = {k: (yte, v["prob"]) for k, v in semua_pred.items()
             if not k.startswith("0b")}
    if kurva:
        ev.plot_roc(kurva, fig / "roc_curves.png")
        ev.plot_pr(kurva, fig / "pr_curves.png")

    best = table.sort_values("TSS", ascending=False).iloc[0]
    if best["model"] in semua_pred:
        ev.plot_confusion(best.to_dict(), fig / "confusion_best.png")
        ev.plot_reliability(yte, semua_pred[best["model"]]["prob"],
                            fig / "reliability.png", name=best["model"])
        ev.plot_threshold_sweep(yte, semua_pred[best["model"]]["prob"],
                                fig / "threshold_sweep.png")

    # Kepentingan fitur konfigurasi terkaya: apakah kolom temporal benar-benar
    # terpakai, atau cuma menambah lebar tabel tanpa dipakai model?
    for label, info in semua_pred.items():
        if not label.startswith("C.") or info["model"] is None:
            continue
        vals, judul = extract_importance(info["model"], info["feats"])
        if not np.any(vals):
            continue
        slug = label.split("| ")[-1].strip().replace(" ", "_").lower()
        path = fig / f"importance_{slug}.png"
        ev.plot_feature_importance(info["feats"], vals, path,
                                   title=f"{judul} -- {label}")
        s = pd.Series(np.abs(vals), index=info["feats"])
        porsi = s[[c for c in info["feats"]
                   if c.endswith("h") and ("_d" in c or "_std" in c)]].sum() / s.sum()
        print(f"\n[{label}] porsi bobot pada fitur temporal: {porsi:.1%}")

    print(f"\n[Grafik] tersimpan di {fig}")
    print(f"[Metrik] tersimpan di {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
