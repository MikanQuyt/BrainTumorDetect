"""
analyze_week2.py - Phan tich ket qua zero-shot SAM vs MedSAM (oracle GT box) cho tuan 2.

    python eval/analyze_week2.py --csv ds4=reports/week2/ds4/oracle_results.csv ds2=reports/week2/ds2/oracle_results.csv --out_dir reports/week2

Dau ra trong --out_dir:
    summary_table.csv       Dice/IoU theo dataset x phuong phap (muc lat va muc ca)
    paired_stats.csv        MedSAM - SAM: delta, khoang tin cay 95% (bootstrap theo ca), ti le thang, Wilcoxon
    strata_blur.csv         Delta theo 3 nhom do ro net bien (mo nhat -> net nhat)
    strata_size.csv         Delta theo 3 nhom kich thuoc u
    oracle_upper_bound.json Moc upper-bound Dice/IoU (hop chuan) cho cac tuan sau
    fig_*.png               Bieu do
    week2_summary.md        Bang tom tat dang markdown de dan vao bao cao
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHODS = [("sam", "SAM"), ("medsam", "MedSAM"), ("boxfill", "Box-fill")]


# ------------------------------------------------------------------ thong ke
def cluster_bootstrap_ci(values, groups, B=5000, seed=0):
    """Khoang tin cay 95% cua trung binh, lay mau lai theo NHOM (ca) de tinh den tuong quan giua cac lat cung ca."""
    codes, uniq = pd.factorize(pd.Series(groups).astype(str))
    sums = np.bincount(codes, weights=np.asarray(values, dtype=float))
    cnts = np.bincount(codes).astype(float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(uniq), size=(B, len(uniq)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(case_delta):
    from scipy.stats import wilcoxon
    d = np.asarray(case_delta, dtype=float)
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan
    try:
        return float(wilcoxon(d).pvalue)
    except Exception:
        return np.nan


def md_table(df, fmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                         else ("" if (isinstance(v, float) and pd.isna(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def p_fmt(p):
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ bang
def summary_table(dfs):
    rows = []
    for ds, df in dfs.items():
        for key, label in METHODS:
            for metric in ("dice", "iou"):
                col = f"{metric}_{key}"
                case = df.groupby("group")[col].mean()
                rows.append(dict(dataset=ds, method=label, metric=metric.upper(),
                                 n_slices=len(df), n_cases=df["group"].nunique(),
                                 slice_mean=df[col].mean(), slice_sd=df[col].std(), slice_median=df[col].median(),
                                 case_mean=case.mean(), case_sd=case.std(), case_median=case.median()))
    return pd.DataFrame(rows)


def paired_stats(dfs):
    rows = []
    for ds, df in dfs.items():
        for metric in ("dice", "iou"):
            d = (df[f"{metric}_medsam"] - df[f"{metric}_sam"]).to_numpy()
            lo, hi = cluster_bootstrap_ci(d, df["group"])
            case_d = pd.Series(d, index=df.index).groupby(df["group"]).mean()
            sd = case_d.std(ddof=1) if len(case_d) > 1 else np.nan
            rows.append(dict(dataset=ds, metric=metric.upper(), n_slices=len(df), n_cases=len(case_d),
                             delta_mean=d.mean(), ci95_low=lo, ci95_high=hi,
                             win_rate_medsam=float((d > 0).mean()), tie_rate=float((d == 0).mean()),
                             case_delta_mean=case_d.mean(),
                             cohen_dz_case=(case_d.mean() / sd) if sd and sd > 0 else np.nan,
                             wilcoxon_p_case=wilcoxon_p(case_d.to_numpy())))
    return pd.DataFrame(rows)


def strata(dfs, by, labels, name):
    rows = []
    for ds, df in dfs.items():
        d = df.dropna(subset=[by]).copy()
        if len(d) < 6:
            continue
        d["stratum"] = pd.qcut(d[by].rank(method="first"), 3, labels=labels)
        d["delta"] = d["dice_medsam"] - d["dice_sam"]
        for lab in labels:
            s = d[d["stratum"] == lab]
            if s.empty:
                continue
            lo, hi = cluster_bootstrap_ci(s["delta"], s["group"])
            rows.append(dict(dataset=ds, strata_by=name, stratum=lab, n=len(s), n_cases=s["group"].nunique(),
                             range_low=s[by].min(), range_high=s[by].max(),
                             dice_sam=s["dice_sam"].mean(), dice_medsam=s["dice_medsam"].mean(),
                             delta=s["delta"].mean(), ci95_low=lo, ci95_high=hi,
                             win_rate_medsam=float((s["delta"] > 0).mean())))
    return pd.DataFrame(rows)


def upper_bound(dfs, run_infos):
    out = {"note": "Upper-bound = Dice/IoU khi prompt la hop chuan (oracle GT box); dung lam moc cho cac tuan sau.",
           "datasets": {}}
    for ds, df in dfs.items():
        entry = {"n_slices": int(len(df)), "n_cases": int(df["group"].nunique()), "methods": {}}
        for key, label in METHODS:
            m = {}
            for metric in ("dice", "iou"):
                col = f"{metric}_{key}"
                case = df.groupby("group")[col].mean()
                m[metric] = dict(slice_mean=float(df[col].mean()), slice_sd=float(df[col].std()),
                                 slice_median=float(df[col].median()), case_mean=float(case.mean()),
                                 case_sd=float(case.std()), case_median=float(case.median()))
            entry["methods"][label] = m
        best = max(["SAM", "MedSAM"], key=lambda k: entry["methods"][k]["dice"]["case_mean"])
        entry["best_model_by_case_mean_dice"] = best
        entry["upper_bound_dice_case_mean"] = entry["methods"][best]["dice"]["case_mean"]
        entry["upper_bound_iou_case_mean"] = entry["methods"][best]["iou"]["case_mean"]
        entry["protocol"] = run_infos.get(ds, {})
        out["datasets"][ds] = entry
    return out


# ------------------------------------------------------------------ hinh
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def fig_boxplots(dfs, out):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for ax, (ds, df) in zip(axes[0], dfs.items()):
        data = [df[f"dice_{k}"].to_numpy() for k, _ in METHODS]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels([lab for _, lab in METHODS])
        ax.set_title(f"{ds}: Dice theo lat (hop chuan)")
        ax.set_ylabel("Dice")
        ax.set_ylim(-0.02, 1.02)
    savefig(out / "fig_dice_boxplot.png")


def fig_delta_vs_sharpness(dfs, out):
    fig, axes = plt.subplots(1, len(dfs), figsize=(5.2 * len(dfs), 4.2), squeeze=False)
    for ax, (ds, df) in zip(axes[0], dfs.items()):
        d = df.dropna(subset=["edge_sharpness"]).copy()
        d["delta"] = d["dice_medsam"] - d["dice_sam"]
        if len(d) >= 20:
            d["bin"] = pd.qcut(d["edge_sharpness"].rank(method="first"), 10, labels=False)
            g = d.groupby("bin").agg(x=("edge_sharpness", "mean"), y=("delta", "mean"),
                                     se=("delta", lambda v: v.std() / np.sqrt(len(v))))
            ax.errorbar(g["x"], g["y"], yerr=1.96 * g["se"], marker="o", capsize=3)
        ax.axhline(0, color="gray", lw=1)
        ax.set_xlabel("Do ro net bien (nho = bien mo)")
        ax.set_ylabel("Delta Dice (MedSAM - SAM)")
        ax.set_title(f"{ds}: Delta theo do ro net (10 nhom phan vi)")
    savefig(out / "fig_delta_vs_sharpness.png")


def fig_strata(blur, out):
    if blur.empty:
        return
    ds_list = blur["dataset"].unique()
    fig, axes = plt.subplots(1, len(ds_list), figsize=(4.6 * len(ds_list), 4.0), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        s = blur[blur["dataset"] == ds]
        x = np.arange(len(s))
        ax.bar(x, s["delta"], yerr=[s["delta"] - s["ci95_low"], s["ci95_high"] - s["delta"]], capsize=4)
        ax.axhline(0, color="gray", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(s["stratum"], rotation=15)
        ax.set_ylabel("Delta Dice (MedSAM - SAM) va CI 95%")
        ax.set_title(f"{ds}: theo nhom do mo cua bien")
    savefig(out / "fig_delta_by_blur_tertile.png")


def overlay(img, mask, color):
    out = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 1)
    return out


def fig_qualitative(ds, df, csv_dir, out, transpose=False):
    d = df.copy()
    d["delta"] = d["dice_medsam"] - d["dice_sam"]
    d = d.sort_values("delta").reset_index(drop=True)
    picks = [("MedSAM hon SAM nhieu nhat", d.iloc[-1]), ("Delta trung vi", d.iloc[len(d) // 2]),
             ("SAM hon MedSAM nhieu nhat", d.iloc[0])]
    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.4 * len(picks)))
    for r_i, (title, r) in enumerate(picks):
        try:
            buf = np.fromfile(r["img_abs"], np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
            gt = cv2.imdecode(np.fromfile(r["mask_abs"], np.uint8), cv2.IMREAD_UNCHANGED)
            gt = (gt[..., 0] if gt.ndim == 3 else gt) > 0
            if transpose:                      # cung he toa do voi mask du doan da luu
                img, gt = np.ascontiguousarray(img.transpose(1, 0, 2)), np.ascontiguousarray(gt.T)
            preds = {}
            for key in ("sam", "medsam"):
                p = csv_dir / "pred" / key / f"{int(r['idx']):06d}.png"
                preds[key] = (cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED) > 0)
        except Exception as e:
            for c in range(4):
                axes[r_i, c].axis("off")
            axes[r_i, 0].set_title(f"{title}: khong doc duoc ({e})", fontsize=8)
            continue
        panels = [(f"{title}\nanh", img), ("Mask chuan", overlay(img, gt, (0, 255, 0))),
                  (f"SAM  Dice={r['dice_sam']:.2f}", overlay(overlay(img, gt, (0, 255, 0)), preds["sam"], (255, 0, 0))),
                  (f"MedSAM  Dice={r['dice_medsam']:.2f}",
                   overlay(overlay(img, gt, (0, 255, 0)), preds["medsam"], (255, 0, 0)))]
        for ax, (t, im) in zip(axes[r_i], panels):
            ax.imshow(im)
            ax.set_title(t, fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{ds}: xanh = mask chuan, do = du doan", fontsize=10)
    savefig(out / f"fig_qualitative_{ds}.png")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="ten=duong_dan_csv (vi du ds4=reports/week2/ds4/oracle_results.csv)")
    ap.add_argument("--out_dir", default="reports/week2")
    ap.add_argument("--allow_dry", action="store_true", help="Cho phep phan tich ket qua DRY RUN (chi de thu)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dfs, run_infos, csv_dirs = {}, {}, {}
    for spec in args.csv:
        name, path = spec.split("=", 1)
        p = Path(path)
        info = {}
        if (p.parent / "run_info.json").exists():
            info = json.load(open(p.parent / "run_info.json", encoding="utf-8"))
        if info.get("dry_run") and not args.allow_dry:
            sys.exit(f"{path} la ket qua DRY RUN (mo hinh gia). Chay that truoc, hoac them --allow_dry de thu.")
        df = pd.read_csv(p)
        need = {"group", "dice_sam", "dice_medsam", "dice_boxfill", "iou_sam", "iou_medsam", "iou_boxfill",
                "edge_sharpness", "tumor_area", "idx", "img_abs", "mask_abs"}
        miss = need - set(df.columns)
        if miss:
            sys.exit(f"{path} thieu cot: {sorted(miss)}")
        dfs[name], run_infos[name], csv_dirs[name] = df, info, p.parent

    summ = summary_table(dfs)
    summ.to_csv(out / "summary_table.csv", index=False)
    paired = paired_stats(dfs)
    paired.to_csv(out / "paired_stats.csv", index=False)
    blur = strata(dfs, "edge_sharpness", ["T1_mo_nhat", "T2_trung_binh", "T3_net_nhat"], "do ro net bien")
    blur.to_csv(out / "strata_blur.csv", index=False)
    size = strata(dfs, "tumor_area", ["T1_nho", "T2_vua", "T3_lon"], "kich thuoc u")
    size.to_csv(out / "strata_size.csv", index=False)
    ub = upper_bound(dfs, run_infos)
    json.dump(ub, open(out / "oracle_upper_bound.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    fig_boxplots(dfs, out)
    fig_delta_vs_sharpness(dfs, out)
    fig_strata(blur, out)
    for ds, df in dfs.items():
        if (csv_dirs[ds] / "pred" / "sam").exists():
            fig_qualitative(ds, df, csv_dirs[ds], out, bool(run_infos[ds].get("transpose")))

    # ---- markdown
    md = ["# Tuan 2: Zero-shot SAM vs MedSAM voi oracle GT box", ""]
    md.append("## 1. Dice / IoU (hop chuan)\n")
    main_t = summ[summ["metric"] == "DICE"][["dataset", "method", "n_slices", "n_cases", "slice_mean", "slice_sd",
                                              "slice_median", "case_mean", "case_sd"]]
    md += ["**Dice**", md_table(main_t), ""]
    iou_t = summ[summ["metric"] == "IOU"][["dataset", "method", "slice_mean", "slice_sd", "slice_median", "case_mean"]]
    md += ["**IoU**", md_table(iou_t), ""]
    md.append("## 2. So sanh cap MedSAM - SAM (bootstrap theo ca, Wilcoxon tren trung binh theo ca)\n")
    pt = paired.copy()
    pt["wilcoxon_p_case"] = pt["wilcoxon_p_case"].map(p_fmt)
    md += [md_table(pt), ""]
    md.append("## 3. Theo do ro net cua bien (T1 = mo nhat)\n")
    md += [md_table(blur) if not blur.empty else "(khong du du lieu)", ""]
    md.append("## 4. Theo kich thuoc u\n")
    md += [md_table(size) if not size.empty else "(khong du du lieu)", ""]
    md.append("## 5. Upper-bound lam moc cho cac tuan sau\n")
    for ds, e in ub["datasets"].items():
        md.append(f"- **{ds}**: mo hinh tot nhat = {e['best_model_by_case_mean_dice']}; "
                  f"Dice (TB theo ca) = {e['upper_bound_dice_case_mean']:.4f}; "
                  f"IoU (TB theo ca) = {e['upper_bound_iou_case_mean']:.4f}")
    md += ["", "_Do ro net bien = gradient Sobel trung binh trong dai hep quanh bien mask chuan tren kenh dau "
               "(da chuan hoa theo khoang cuong do); gia tri nho = bien mo._"]
    (out / "week2_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nDa luu ket qua vao: {out.resolve()}")


if __name__ == "__main__":
    main()
