"""Backtest report generation – markdown and JSON output."""

import json
from pathlib import Path


def generate_report(results: dict, output_dir: str = ".") -> str:
    """Generate a markdown report from backtest results.

    Also saves a JSON report to backtest_report.json.

    Args:
        results: Dict with per-crisis results and summary stats.
        output_dir: Directory to write the JSON report to.

    Returns:
        Markdown report string.
    """
    lines: list[str] = []
    lines.append("# Backtest Report")
    lines.append("")

    # Summary section
    total_signals = sum(
        r.get("num_signals", 0) for r in results.get("crises", {}).values()
    )
    hit_rates = [
        r["hit_rate"]
        for r in results.get("crises", {}).values()
        if r.get("hit_rate") is not None
    ]
    overall_hit_rate = (
        sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    )

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total signals generated**: {total_signals}")
    lines.append(f"- **Overall hit rate**: {overall_hit_rate:.2%}")
    lines.append("")

    # Per-crisis section
    lines.append("## Per-Crisis Results")
    lines.append("")

    for crisis_name, crisis_result in results.get("crises", {}).items():
        lines.append(f"### {crisis_name}")
        lines.append("")
        lines.append(
            f"- Signals generated: {crisis_result.get('num_signals', 0)}"
        )
        hit_rate = crisis_result.get("hit_rate")
        lines.append(
            f"- Hit rate: {hit_rate:.2%}" if hit_rate is not None else "- Hit rate: N/A"
        )
        fp_rate = crisis_result.get("false_positive_rate")
        lines.append(
            f"- False positive rate: {fp_rate:.2%}"
            if fp_rate is not None
            else "- False positive rate: N/A"
        )
        regime_count = crisis_result.get("regime_predictions_count", 0)
        lines.append(f"- Regime predictions: {regime_count}")
        lines.append("")

    report_md = "\n".join(lines)

    # Save JSON report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_data = {
        "total_signals": total_signals,
        "overall_hit_rate": overall_hit_rate,
        "crises": results.get("crises", {}),
    }
    json_path = output_path / "backtest_report.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    return report_md
