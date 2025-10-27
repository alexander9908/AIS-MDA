from pathlib import Path
import json
import statistics

def load_all(metrics_dir="metrics"):
    rows = []
    for p in Path(metrics_dir).glob("*.json"):
        with open(p) as f:
            d = json.load(f)
            rows.append(d)
    return rows


def fmt(v, n=3):
    """Format float or return '-'."""
    if v is None or v == "-":
        return "-"
    try:
        return f"{v:.{n}f}"
    except Exception:
        return "-"


def main():
    rows = load_all()
    if not rows:
        print("No metrics/*.json found. Run training/evaluation first.")
        return

    # Sort by task then model for consistent output
    rows = sorted(rows, key=lambda r: (r.get("task", ""), r.get("model", "")))

    print("\n" + "═" * 80)
    print("📊  MODEL PERFORMANCE SUMMARY")
    print("═" * 80 + "\n")
    print("| Task | Model | ADE | FDE | MAE | MAPE | P95 | AUROC | AUPRC | Checkpoint |")
    print("|:-----|:------|----:|----:|----:|-----:|----:|------:|------:|:------------|")

    # Track stats for averages
    metrics_by_task = {}

    for r in rows:
        task = r.get("task", "-")
        model = r.get("model", "-")
        ckpt = Path(r.get("ckpt", "")).name

        ade = r.get("ade")
        fde = r.get("fde")
        mae = r.get("mae")
        mape = r.get("mape")
        p95 = r.get("p95")
        auroc = r.get("auroc")
        auprc = r.get("auprc")

        # Store for averages
        if task not in metrics_by_task:
            metrics_by_task[task] = {"ade": [], "fde": [], "mae": [], "mape": [], "p95": [], "auroc": [], "auprc": []}
        for k in ["ade", "fde", "mae", "mape", "p95", "auroc", "auprc"]:
            if isinstance(r.get(k), (int, float)):
                metrics_by_task[task][k].append(r[k])

        # Add visual indicators for key metrics
        tag = ""
        if task == "trajectory":
            tag = "📈"
        elif task == "eta":
            tag = "⏱️"
        elif task == "anomaly":
            tag = "⚠️"

        print(
            f"| {tag} {task:<10} | {model:<6} | "
            f"{fmt(ade):>6} | {fmt(fde):>6} | {fmt(mae):>6} | {fmt(mape):>6} | "
            f"{fmt(p95):>6} | {fmt(auroc):>6} | {fmt(auprc):>6} | {ckpt:<14} |"
        )

    print("\n" + "─" * 80)
    print("📊  AVERAGES BY TASK")
    print("─" * 80)
    print("| Task | ADE | FDE | MAE | MAPE | P95 | AUROC | AUPRC |")
    print("|:-----|----:|----:|----:|-----:|----:|------:|------:|")
    for task, vals in metrics_by_task.items():
        print(
            f"| {task:<10} | "
            f"{fmt(statistics.mean(vals['ade'])) if vals['ade'] else '-':>6} | "
            f"{fmt(statistics.mean(vals['fde'])) if vals['fde'] else '-':>6} | "
            f"{fmt(statistics.mean(vals['mae'])) if vals['mae'] else '-':>6} | "
            f"{fmt(statistics.mean(vals['mape'])) if vals['mape'] else '-':>6} | "
            f"{fmt(statistics.mean(vals['p95'])) if vals['p95'] else '-':>6} | "
            f"{fmt(statistics.mean(vals['auroc'])) if vals['auroc'] else '-':>6} | "
            f"{fmt(statistics.mean(vals['auprc'])) if vals['auprc'] else '-':>6} |"
        )

    print("\n✅ Saved metrics are ready for markdown copy-paste or report inclusion.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()