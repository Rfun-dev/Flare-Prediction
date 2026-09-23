"""Matriks evaluasi untuk prakiraan flare.

Akurasi TIDAK BOLEH dipakai sebagai metrik utama di sini. Dengan basis rate
kelas-M sekitar 1-3%, model yang selalu menjawab "tidak ada flare" mencapai
akurasi ~98% dan TSS = 0 -- sempurna secara akurasi, tanpa guna sama sekali.
Metrik utama domain space weather adalah TSS dan HSS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ==============================================================================
# Palet & gaya plot (light mode, mengikuti reference palette dataviz)
# ==============================================================================
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style_axes(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(C_SURFACE)
    ax.figure.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=C_MUTED, labelsize=9, length=3, width=1.0)
    ax.grid(True, color=C_GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_INK2, fontsize=10)
    if title:
        ax.set_title(title, color=C_INK, fontsize=12, fontweight="bold",
                     loc="left", pad=12)


# ==============================================================================
# 1. Metrik deterministik (satu ambang)
# ==============================================================================

def contingency(y_true, y_pred) -> dict[str, int]:
    """Tabel kontingensi 2x2. Istilah domain: TP=hit, FP=false alarm,
    FN=miss, TN=correct negative."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "TP": int(np.sum((y_true == 1) & (y_pred == 1))),
        "FP": int(np.sum((y_true == 0) & (y_pred == 1))),
        "FN": int(np.sum((y_true == 1) & (y_pred == 0))),
        "TN": int(np.sum((y_true == 0) & (y_pred == 0))),
    }


def _safe(num, den, default=0.0):
    return float(num) / float(den) if den else default


def deterministic_metrics(y_true, y_pred) -> dict[str, float]:
    """Seluruh skor berbasis tabel kontingensi."""
    c = contingency(y_true, y_pred)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    n = tp + fp + fn + tn

    pod = _safe(tp, tp + fn)          # hit rate / recall / sensitivitas
    pofd = _safe(fp, fp + tn)         # false alarm RATE (sumbu x kurva ROC)
    far = _safe(fp, tp + fp)          # false alarm RATIO (bukan hal yang sama!)
    precision = 1.0 - far if (tp + fp) else 0.0
    specificity = _safe(tn, tn + fp)

    tss = pod - pofd                  # Hanssen-Kuipers; 0 = tanpa keterampilan

    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = _safe(2 * (tp * tn - fp * fn), hss_den)

    # Gilbert Skill Score (ETS): hit yang diperoleh secara kebetulan dikoreksi
    hits_random = _safe((tp + fp) * (tp + fn), n)
    gss = _safe(tp - hits_random, tp + fp + fn - hits_random)

    csi = _safe(tp, tp + fp + fn)
    bias = _safe(tp + fp, tp + fn)    # >1 = terlalu sering memberi peringatan
    f1 = _safe(2 * tp, 2 * tp + fp + fn)

    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe(tp * tn - fp * fn, mcc_den)

    return {
        **c,
        "N": n,
        "base_rate": _safe(tp + fn, n),
        "POD": pod,
        "POFD": pofd,
        "FAR": far,
        "Precision": precision,
        "Recall": pod,
        "Specificity": specificity,
        "F1": f1,
        "CSI": csi,
        "FrequencyBias": bias,
        "TSS": tss,
        "HSS": hss,
        "GSS": gss,
        "MCC": mcc,
        "Accuracy": _safe(tp + tn, n),
    }


# ==============================================================================
# 2. Metrik probabilistik
# ==============================================================================

def probabilistic_metrics(y_true, y_prob, climatology: float | None = None) -> dict:
    """ROC-AUC, PR-AUC, Brier, dan Brier Skill Score.

    BSS membandingkan model terhadap prakiraan klimatologi (selalu menebak
    basis rate). BSS <= 0 berarti model kalah dari tebakan konstan.

    Catatan penting: Brier dan BSS hanya bermakna kalau probabilitasnya
    terkalibrasi. Model dengan `class_weight="balanced"` menghasilkan
    probabilitas yang sengaja digeser -- lihat `models.ProbabilityCalibrator`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"ROC_AUC": np.nan, "PR_AUC": np.nan, "Brier": np.nan, "BSS": np.nan}

    clim = float(np.mean(y_true)) if climatology is None else float(climatology)
    bs = brier_score_loss(y_true, y_prob)
    bs_ref = float(np.mean((y_true - clim) ** 2))
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "Brier": float(bs),
        "BSS": _safe(bs_ref - bs, bs_ref),
        "Climatology": clim,
    }


# ==============================================================================
# 3. Pemilihan ambang
# ==============================================================================

def threshold_sweep(y_true, y_prob, n_steps: int = 200) -> pd.DataFrame:
    """Skor deterministik pada rentang ambang probabilitas."""
    thresholds = np.unique(np.round(np.linspace(0.001, 0.999, n_steps), 4))
    rows = []
    for thr in thresholds:
        m = deterministic_metrics(y_true, (np.asarray(y_prob) >= thr).astype(int))
        m["threshold"] = float(thr)
        rows.append(m)
    return pd.DataFrame(rows)


def best_threshold(y_true, y_prob, metric: str = "TSS") -> tuple[float, float]:
    """Ambang yang memaksimalkan metrik pilihan.

    HARUS dicari pada set VALIDASI, lalu dibekukan sebelum menyentuh set test.
    Ambang default 0.5 hampir selalu buruk pada data yang sangat tidak seimbang.
    """
    sweep = threshold_sweep(y_true, y_prob)
    i = int(sweep[metric].idxmax())
    return float(sweep.loc[i, "threshold"]), float(sweep.loc[i, metric])


# ==============================================================================
# 4. Selang kepercayaan (bootstrap berbasis blok AR)
# ==============================================================================

def bootstrap_ci(y_true, y_pred, groups=None, metric: str = "TSS",
                 n_boot: int = 500, alpha: float = 0.05,
                 random_state: int = 42) -> tuple[float, float]:
    """CI persentil untuk metrik deterministik.

    Resampling dilakukan per HARP (blok), bukan per baris: baris-baris dari AR
    yang sama sangat berkorelasi, sehingga bootstrap per baris menghasilkan
    selang yang terlalu sempit (over-confident).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if groups is None:
        idx_pool = [np.arange(len(y_true))]
    else:
        groups = np.asarray(groups)
        idx_pool = [np.where(groups == g)[0] for g in np.unique(groups)]

    scores = []
    n_blocks = len(idx_pool)
    for _ in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([idx_pool[p] for p in pick])
        s = deterministic_metrics(y_true[idx], y_pred[idx])[metric]
        scores.append(s)
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ==============================================================================
# 5. Laporan gabungan
# ==============================================================================

def evaluate(y_true, y_prob, threshold: float = 0.5, groups=None,
             name: str = "model", climatology: float | None = None,
             n_boot: int = 300) -> dict:
    """Satu panggilan -> seluruh matriks evaluasi untuk satu model."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    report = {"model": name, "threshold": float(threshold)}
    report.update(deterministic_metrics(y_true, y_pred))
    report.update(probabilistic_metrics(y_true, y_prob, climatology))

    if n_boot:
        for m in ("TSS", "HSS"):
            lo, hi = bootstrap_ci(y_true, y_pred, groups, metric=m, n_boot=n_boot)
            report[f"{m}_lo"] = lo
            report[f"{m}_hi"] = hi
    return report


METRIC_ORDER = [
    "model", "threshold", "N", "base_rate",
    "TP", "FP", "FN", "TN",
    "TSS", "TSS_lo", "TSS_hi", "HSS", "HSS_lo", "HSS_hi", "GSS", "MCC",
    "POD", "POFD", "FAR", "Precision", "F1", "CSI", "FrequencyBias",
    "ROC_AUC", "PR_AUC", "Brier", "BSS", "Accuracy",
]


def report_table(reports: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(reports)
    cols = [c for c in METRIC_ORDER if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


def print_report(rep: dict) -> None:
    line = "=" * 68
    print(f"\n{line}")
    print(f"  {rep['model']}   (ambang = {rep['threshold']:.3f}, N = {rep['N']:,})")
    print(line)
    print(f"  Tabel kontingensi   hit={rep['TP']:<6} miss={rep['FN']:<6} "
          f"false alarm={rep['FP']:<7} correct neg={rep['TN']:,}")
    print(f"  {'-' * 64}")
    ci_t = f"  [{rep['TSS_lo']:+.3f}, {rep['TSS_hi']:+.3f}]" if "TSS_lo" in rep else ""
    ci_h = f"  [{rep['HSS_lo']:+.3f}, {rep['HSS_hi']:+.3f}]" if "HSS_lo" in rep else ""
    print(f"  TSS (utama)      {rep['TSS']:+.4f}{ci_t}")
    print(f"  HSS              {rep['HSS']:+.4f}{ci_h}")
    print(f"  GSS / ETS        {rep['GSS']:+.4f}")
    print(f"  MCC              {rep['MCC']:+.4f}")
    print(f"  {'-' * 64}")
    print(f"  POD (recall)     {rep['POD']:.4f}      POFD  {rep['POFD']:.4f}")
    print(f"  Precision        {rep['Precision']:.4f}      FAR   {rep['FAR']:.4f}")
    print(f"  F1               {rep['F1']:.4f}      CSI   {rep['CSI']:.4f}")
    print(f"  Frequency bias   {rep['FrequencyBias']:.4f}  (1.0 = jumlah peringatan pas)")
    print(f"  {'-' * 64}")
    if not np.isnan(rep.get("ROC_AUC", np.nan)):
        print(f"  ROC-AUC          {rep['ROC_AUC']:.4f}      PR-AUC  {rep['PR_AUC']:.4f}")
        print(f"  Brier            {rep['Brier']:.5f}     BSS     {rep['BSS']:+.4f}")
    print(f"  Accuracy         {rep['Accuracy']:.4f}  (menyesatkan -- basis rate "
          f"{rep['base_rate']:.2%})")
    print(line)


def save_reports(reports: list[dict], out_dir: Path, stem: str = "metrics") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = report_table(reports)
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, default=float)
    return csv_path


# ==============================================================================
# 6. Grafik diagnostik
# ==============================================================================

def plot_roc(curves: dict[str, tuple], out_path: Path):
    """curves: {nama: (y_true, y_prob)}"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=140)
    ax.plot([0, 1], [0, 1], color=C_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
            zorder=1, label="tanpa keterampilan")
    for i, (name, (yt, yp)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, color=C_SERIES[i % len(C_SERIES)], linewidth=2.0,
                solid_capstyle="round", zorder=3 + i, label=f"{name}  (AUC {auc:.3f})")
    _style_axes(ax, "POFD  (false alarm rate)", "POD  (hit rate)", "Kurva ROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C_INK2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=C_SURFACE)
    plt.close(fig)


def plot_pr(curves: dict[str, tuple], out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=140)
    base = None
    for i, (name, (yt, yp)) in enumerate(curves.items()):
        base = float(np.mean(np.asarray(yt)))
        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ax.plot(rec, prec, color=C_SERIES[i % len(C_SERIES)], linewidth=2.0,
                solid_capstyle="round", zorder=3 + i, label=f"{name}  (AP {ap:.3f})")
    if base is not None:
        ax.axhline(base, color=C_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=1, label=f"klimatologi ({base:.3f})")
    _style_axes(ax, "Recall  (POD)", "Precision", "Kurva Precision-Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C_INK2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=C_SURFACE)
    plt.close(fig)


def plot_reliability(y_true, y_prob, out_path: Path, n_bins: int = 10,
                     name: str = "model"):
    """Diagram reliabilitas: apakah 'probabilitas 30%' benar-benar terjadi 30%?"""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)

    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5:
            xs.append(y_prob[m].mean())
            ys.append(y_true[m].mean())
            ns.append(int(m.sum()))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.4, 6.0), dpi=140,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.32})

    ax.plot([0, 1], [0, 1], color=C_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
            zorder=1, label="kalibrasi sempurna")
    ax.plot(xs, ys, color=C_SERIES[0], linewidth=2.0, marker="o", markersize=7,
            markeredgecolor=C_SURFACE, markeredgewidth=2.0, zorder=3, label=name)
    ax.axhline(float(y_true.mean()), color=C_SERIES[1], linewidth=1.5,
               linestyle=(0, (2, 2)), zorder=2, label="klimatologi")
    _style_axes(ax, "", "Frekuensi teramati", "Diagram reliabilitas (kalibrasi)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C_INK2)

    ax2.bar(xs, ns, width=0.8 / n_bins, color=C_SERIES[0],
            edgecolor=C_SURFACE, linewidth=1.5)
    _style_axes(ax2, "Probabilitas prakiraan", "Jumlah sampel")
    ax2.set_xlim(0, 1)
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=C_SURFACE)
    plt.close(fig)


def plot_confusion(rep: dict, out_path: Path):
    """Peta panas 2x2, dinormalkan per baris (per kelas sebenarnya)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    counts = np.array([[rep["TN"], rep["FP"]], [rep["FN"], rep["TP"]]], dtype=float)
    row_sum = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=140)
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)

    labels = [["correct negative", "false alarm"], ["miss", "hit"]]
    for i in range(2):
        for j in range(2):
            ink = "#ffffff" if frac[i, j] > 0.55 else C_INK
            ax.text(j, i - 0.13, f"{int(counts[i, j]):,}", ha="center", va="center",
                    color=ink, fontsize=15, fontweight="bold")
            ax.text(j, i + 0.14, f"{frac[i, j]:.1%}  {labels[i][j]}", ha="center",
                    va="center", color=ink, fontsize=9)

    ax.set_xticks([0, 1], ["prakiraan: tenang", "prakiraan: flare"])
    ax.set_yticks([0, 1], ["nyata: tenang", "nyata: flare"])
    ax.tick_params(colors=C_INK2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.set_title(f"Tabel kontingensi -- {rep['model']}", color=C_INK, fontsize=12,
                 fontweight="bold", loc="left", pad=12)
    fig.set_facecolor(C_SURFACE)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=C_SURFACE)
    plt.close(fig)


def plot_threshold_sweep(y_true, y_prob, out_path: Path,
                         metrics=("TSS", "HSS", "F1")):
    """Skor sebagai fungsi ambang -- memperlihatkan bahwa 0.5 jarang optimal."""
    import matplotlib.pyplot as plt

    sweep = threshold_sweep(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=140)
    for i, m in enumerate(metrics):
        j = int(sweep[m].idxmax())
        # Nilai maksimum ditaruh di LEGENDA, bukan sebagai anotasi di dekat
        # puncaknya. Beberapa metrik kerap memuncak pada ambang yang sama
        # (HSS dan F1 hampir selalu begitu), sehingga anotasi yang ditambatkan
        # ke titik akan saling menimpa berapa pun offset yang dipilih.
        ax.plot(sweep["threshold"], sweep[m], color=C_SERIES[i % len(C_SERIES)],
                linewidth=2.0, solid_capstyle="round", zorder=3 + i,
                label=f"{m}   maks {sweep.loc[j, m]:.3f} @ {sweep.loc[j, 'threshold']:.2f}")
        ax.plot(sweep.loc[j, "threshold"], sweep.loc[j, m], marker="o", markersize=8,
                color=C_SERIES[i % len(C_SERIES)], markeredgecolor=C_SURFACE,
                markeredgewidth=2.0, zorder=10)
    ax.axvline(0.5, color=C_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    _style_axes(ax, "Ambang probabilitas", "Skor", "Skor terhadap ambang keputusan")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.annotate("ambang default 0.5", (0.5, ax.get_ylim()[1] * 0.03),
                textcoords="offset points", xytext=(6, 0), color=C_MUTED, fontsize=8.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(C_INK2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=C_SURFACE)
    plt.close(fig)


def plot_model_comparison(df_reports: pd.DataFrame, out_path: Path,
                          metric: str = "TSS"):
    """Perbandingan model pada satu metrik, dengan galat bootstrap bila ada."""
    import matplotlib.pyplot as plt

    d = df_reports.sort_values(metric)
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(d) + 1.8), dpi=140)
    ypos = np.arange(len(d))

    err = None
    if f"{metric}_lo" in d.columns:
        err = np.vstack([d[metric] - d[f"{metric}_lo"], d[f"{metric}_hi"] - d[metric]])
        err = np.clip(err, 0, None)

    ax.barh(ypos, d[metric], height=0.6, color=C_SERIES[0],
            edgecolor=C_SURFACE, linewidth=2.0, zorder=3)
    if err is not None:
        ax.errorbar(d[metric], ypos, xerr=err, fmt="none", ecolor=C_INK2,
                    elinewidth=1.2, capsize=4, zorder=5)
    # Label ditaruh di kanan UJUNG ERROR BAR, bukan ujung batang, supaya tidak
    # menimpa whisker.
    hi = d[f"{metric}_hi"] if f"{metric}_hi" in d.columns else d[metric]
    right = np.maximum(d[metric].to_numpy(), hi.to_numpy())
    span = max(float(right.max()), 0.1)
    for y, v, r in zip(ypos, d[metric], right):
        ax.text(r + 0.025 * span, y, f"{v:.3f}", va="center",
                color=C_INK2, fontsize=9.5)
    ax.set_xlim(min(0.0, float(d[metric].min()) * 1.1), span * 1.22)

    ax.axvline(0, color=C_AXIS, linewidth=1.2, zorder=2)
    ax.set_yticks(ypos, d["model"])
    _style_axes(ax, metric, "", f"Perbandingan model -- {metric} (set test)")
    ax.tick_params(axis="y", colors=C_INK, labelsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=C_SURFACE)
    plt.close(fig)


def plot_feature_importance(names, values, out_path: Path, top_n: int = 15,
                            title: str = "Kepentingan fitur"):
    import matplotlib.pyplot as plt

    s = pd.Series(np.asarray(values, dtype=float), index=list(names))
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(s) + 1.7), dpi=140)
    colors = [C_SERIES[0] if v >= 0 else C_SERIES[1] for v in s.values]
    ax.barh(np.arange(len(s)), s.values, height=0.68, color=colors,
            edgecolor=C_SURFACE, linewidth=2.0, zorder=3)
    ax.axvline(0, color=C_AXIS, linewidth=1.2, zorder=2)
    ax.set_yticks(np.arange(len(s)), s.index)
    _style_axes(ax, "", "", title)
    ax.tick_params(axis="y", colors=C_INK, labelsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=C_SURFACE)
    plt.close(fig)
