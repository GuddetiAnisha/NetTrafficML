"""Create a concise Markdown experiment report from metrics.json."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
metrics_path = ROOT / "outputs" / "metrics.json"
out = ROOT / "outputs" / "experiment_report.md"

def main():
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    lines = [
        "# NetTrafficML Experiment Report",
        "",
        f"- Rows: {m['dataset']['rows']:,}",
        f"- Congestion positive rate: {m['dataset']['positive_rate']:.3f}",
        f"- Best classifier: {m['best_classifier']}",
        "",
        "## Classification",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in m["classification"].items():
        lines.append(
            f"| {name} | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} | "
            f"{v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} |"
        )
    r = m["regression"]["point"]
    p = m["regression"]["probabilistic"]
    lines += [
        "", "## Regression", "",
        f"- MAE: {r['mae']:.4f}",
        f"- RMSE: {r['rmse']:.4f}",
        f"- R2: {r['r2']:.4f}",
        f"- 10th-90th interval empirical coverage: {p['empirical_coverage']:.4f}",
        "", "## Geographical generalization", "",
        "| Held-out region | ROC-AUC | PR-AUC | Samples |",
        "|---|---:|---:|---:|",
    ]
    for region, v in m["generalization"].items():
        lines.append(
            f"| {region} | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} | {v['samples']} |"
        )
    lines += [
        "", "## Feature selection", "",
        *[
            f"- {x['feature']}: {x['score']:.5f}"
            for x in m["feature_selection"]["mutual_information"]
        ],
        "",
        f"Best K-Means k: {m['clustering']['best_k']}",
        f"PCA components for 95% variance: "
        f"{m['dimensionality_reduction']['components_for_95pct_variance']}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
