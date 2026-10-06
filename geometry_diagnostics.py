"""Compare two pose metric sets. Never writes settings.json.

Hold a correct pose and a confused pose, then inspect which relative
metrics separate them. Production thresholds are not changed from here.
"""
import statistics

from geometry_engine import relative_metrics


def metrics_from_landmarks(landmarks):
    return relative_metrics(landmarks)


def _summary(values):
    numbers = [float(item) for item in values]
    if not numbers:
        return None
    ordered = sorted(numbers)
    return {
        "min": ordered[0],
        "median": float(statistics.median(ordered)),
        "max": ordered[-1],
        "n": len(ordered),
    }


def compare_metric_sets(correct_rows, impostor_rows):
    """Return per-metric summaries and discriminators with a gap (no overlap).

    Each row is a dict from relative_metrics() or metrics_from_landmarks().
    """
    keys = []
    for row in list(correct_rows or []) + list(impostor_rows or []):
        for key, value in (row or {}).items():
            if isinstance(value, (int, float)) and key not in keys:
                keys.append(key)
    report = {"metrics": {}, "best_discriminators": []}
    for key in keys:
        good = [float(row[key]) for row in (correct_rows or []) if key in row]
        bad = [float(row[key]) for row in (impostor_rows or []) if key in row]
        left, right = _summary(good), _summary(bad)
        report["metrics"][key] = {"correct": left, "impostor": right}
        if left is None or right is None:
            continue
        if left["max"] < right["min"]:
            gap = right["min"] - left["max"]
            report["best_discriminators"].append({
                "metric": key,
                "gap": gap,
                "correct": f"{left['min']:.3f}–{left['max']:.3f}",
                "impostor": f"{right['min']:.3f}–{right['max']:.3f}",
            })
        elif right["max"] < left["min"]:
            gap = left["min"] - right["max"]
            report["best_discriminators"].append({
                "metric": key,
                "gap": gap,
                "correct": f"{left['min']:.3f}–{left['max']:.3f}",
                "impostor": f"{right['min']:.3f}–{right['max']:.3f}",
            })
    report["best_discriminators"].sort(key=lambda item: item["gap"], reverse=True)
    return report


def format_discriminators(report, limit=8):
    lines = ["BEST DISCRIMINATORS"]
    rows = (report or {}).get("best_discriminators") or []
    if not rows:
        lines.append("(no non-overlapping metric)")
        return "\n".join(lines)
    for item in rows[:limit]:
        lines.append(item["metric"])
        lines.append(f"  correct: {item['correct']}")
        lines.append(f"  impostor: {item['impostor']}")
        lines.append(f"  gap: {item['gap']:.3f}")
    return "\n".join(lines)
