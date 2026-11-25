"""Validate the held-out ledger and derive figures and paired statistics."""

from collections import Counter
import csv
from itertools import product
import json

import numpy as np

from style_judges.inference import get_manifest, open_results, require_complete, row_key, verdict_logits
from style_judges.protocol import ORDERS, PROMPTS, SEED, STYLES, read_json, write_json
from style_judges.sources import MODELS


EDITS = STYLES[1:]
BOOTSTRAPS = 10_000


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def cluster_draws(sources):
    rng = np.random.default_rng(SEED)
    by_stratum = [np.flatnonzero(sources == source) for source in sorted(set(sources))]
    return np.concatenate(
        [rng.choice(group, size=(BOOTSTRAPS, len(group)), replace=True) for group in by_stratum],
        axis=1,
    )


def collect_statistics(rows, verdicts):
    sources = np.array([row["source"] for row in rows])
    verified = np.array([row["reference"] == "fixed" for row in rows])
    masks = {"all": np.ones(len(rows), dtype=bool), "verified": verified}
    masks.update({source: sources == source for source in sorted(set(sources))})
    draws = {name: cluster_draws(sources[mask]) for name, mask in masks.items() if mask.any()}
    metrics = []
    transitions = []

    def add(name, model, prompt, variant, values, subset="all"):
        sampled = np.asarray(values, dtype=float)[masks[subset]]
        if not len(sampled):
            return
        flat = sampled.reshape(len(sampled), -1)
        means = flat.mean(axis=1)
        ci = np.percentile(means[draws[subset]].mean(axis=1), [2.5, 97.5])
        metrics.append({
            "metric": name, "model": model, "prompt": prompt, "variant": variant,
            "subset": subset, "estimate": float(np.mean(means)),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]), "questions": len(sampled),
            "numerator": int(flat.sum()), "denominator": flat.size,
        })

    for model, prompt in product(MODELS, PROMPTS):
        baseline = verdicts[model, prompt, "original"]
        repeat = verdicts[model, prompt, "repeat"]
        add("repeat_consistency", model, prompt, "original", baseline == repeat)
        add("baseline_flip", model, prompt, "original", baseline != repeat)
        for variant in STYLES:
            edited = verdicts[model, prompt, variant]
            add("tie_rate", model, prompt, variant, edited == "tie")
            add("order_consistency", model, prompt, variant, edited[:, 0] == edited[:, 1])
            add("reference_accuracy", model, prompt, variant, edited == "fixed", "verified")
            add("target_selection", model, prompt, variant, edited == "target")
            if variant == "original":
                continue
            flip = baseline != edited
            add("flip_rate", model, prompt, variant, flip)
            add("flip_rate", model, prompt, variant, flip, "verified")
            for source in sorted(set(sources)):
                add("flip_rate", model, prompt, variant, flip, source)
            change = (edited == "target").astype(int) - (baseline == "target").astype(int)
            add("target_selection_change", model, prompt, variant, change)
            change = (edited == "fixed").astype(int) - (baseline == "fixed").astype(int)
            add("reference_accuracy_change", model, prompt, variant, change, "verified")
            for subset in ("all", "verified"):
                counts = Counter(zip(baseline[masks[subset]].ravel(), edited[masks[subset]].ravel(), strict=True))
                for previous, current in product(("fixed", "target", "tie"), repeat=2):
                    transitions.append({
                        "model": model, "prompt": prompt, "variant": variant,
                        "subset": subset, "from": previous, "to": current,
                        "count": counts[previous, current],
                    })

    for model, variant in product(MODELS, STYLES):
        brief = verdicts[model, "brief", variant]
        rubric = verdicts[model, "rubric", variant]
        change = (rubric == "fixed").astype(int) - (brief == "fixed").astype(int)
        add("rubric_accuracy_effect", model, "rubric_minus_brief", variant, change, "verified")
        if variant in EDITS:
            brief_flip = verdicts[model, "brief", "original"] != brief
            rubric_flip = verdicts[model, "rubric", "original"] != rubric
            add("rubric_flip_effect", model, "rubric_minus_brief", variant,
                rubric_flip.astype(int) - brief_flip.astype(int))

    for prompt, variant in product(PROMPTS, STYLES):
        small = verdicts["Qwen3-0.6B", prompt, variant]
        large = verdicts["Qwen3-4B", prompt, variant]
        change = (large == "fixed").astype(int) - (small == "fixed").astype(int)
        add("size_accuracy_effect", "4B_minus_0.6B", prompt, variant, change, "verified")
        if variant in EDITS:
            small_flip = verdicts["Qwen3-0.6B", prompt, "original"] != small
            large_flip = verdicts["Qwen3-4B", prompt, "original"] != large
            add("size_flip_effect", "4B_minus_0.6B", prompt, variant,
                large_flip.astype(int) - small_flip.astype(int))
    return metrics, transitions


def collect_timing(root, records, precision):
    timing = []
    for model, judgments in records.items():
        metadata = read_json(root / "runs" / f"heldout-{precision}-{model}.json")
        if metadata["completed"] != len(judgments):
            raise RuntimeError(f"Incomplete memory telemetry for {model}")
        for variant in ("all", *STYLES):
            rows = [row for row in judgments if row["variant"] != "repeat"
                    and (variant == "all" or row["variant"] == variant)]
            samples = [row["latency_ms"] for row in rows]
            timing.append({
                "model": model, "variant": variant, "decisions": len(samples),
                "median_latency_ms": float(np.median(samples)),
                "p95_latency_ms": float(np.percentile(samples, 95)),
                "median_input_tokens": float(np.median([row["input_tokens"] for row in rows])),
                "max_input_tokens": max(row["input_tokens"] for row in rows),
                "peak_inference_allocated_gib": metadata["inference_peak_allocated_bytes"] / 2**30,
                "peak_inference_reserved_gib": metadata["inference_peak_reserved_bytes"] / 2**30,
                "peak_load_allocated_gib": metadata["load_peak_allocated_bytes"] / 2**30,
                "peak_load_reserved_gib": metadata["load_peak_reserved_bytes"] / 2**30,
            })
    return timing


def examples(rows, verdicts):
    selected = []
    chosen = set()
    for variant in EDITS:
        candidates = []
        for idx, row in enumerate(rows):
            if row["pair_id"] in chosen:
                continue
            changes = []
            for model, prompt in product(MODELS, PROMPTS):
                before = verdicts[model, prompt, "original"][idx]
                after = verdicts[model, prompt, variant][idx]
                for order, previous, current in zip(ORDERS, before, after, strict=True):
                    if current != previous:
                        changes.append({
                            "model": model, "prompt": prompt, "order": order,
                            "before": str(previous), "after": str(current),
                        })
            if changes:
                harmful = [change for change in changes if row["reference"] == "fixed"
                           and change["before"] == "fixed" and change["after"] != "fixed"]
                moves = sum(change["after"] == "target" for change in changes)
                rank = (-len({change["model"] for change in harmful}), -len(harmful),
                        -len({change["model"] for change in changes}), -len(changes), -moves,
                        row["pair_id"])
                candidates.append((rank, row, changes, len(harmful)))
        if candidates:
            _, row, changes, harms = min(candidates, key=lambda candidate: candidate[0])
            selected.append({"variant": variant, "row": row, "changes": changes,
                             "verified_harmful_changes": harms})
            chosen.add(row["pair_id"])
    return selected


def plot_results(root, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = root / "figures"
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.spines.right": False, "axes.spines.top": False,
        "axes.labelcolor": "#202b37", "text.color": "#17212a", "pdf.fonttype": 42,
    })
    palettes = {"brief": "#117882", "rubric": "#ae462a"}
    markers = {"brief": "o", "rubric": "s"}
    lookup = {(m["metric"], m["model"], m["prompt"], m["variant"], m["subset"]): m for m in metrics}

    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True, sharey=True, layout="constrained")
    max_ci = max(m["ci_high"] for m in metrics if m["metric"] == "flip_rate" and m["subset"] == "all")
    short_labels = [model.removeprefix("Qwen3-") for model in MODELS]
    for axis, variant in zip(axes, EDITS, strict=True):
        for pos, model in enumerate(MODELS):
            for prompt, offset in (("brief", -.12), ("rubric", .12)):
                m = lookup["flip_rate", model, prompt, variant, "all"]
                x = m["estimate"] * 100
                error = [[max(0, x - m["ci_low"] * 100)], [max(0, m["ci_high"] * 100 - x)]]
                label = prompt.capitalize() if pos == 0 else None
                axis.errorbar(x, pos + offset, xerr=error, fmt=markers[prompt], color=palettes[prompt],
                              label=label, capsize=2.6, markersize=4.5, elinewidth=1.1)
        axis.set_title(variant.capitalize(), fontweight="bold")
        axis.grid(axis="x", alpha=.25)
        axis.set_yticks(range(len(MODELS)), short_labels)
        axis.set_xlim(-1, max(12, max_ci * 110))
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Qwen3 model size")
    axes[1].set_xlabel("Held-out preferences that changed (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, fontsize=8)
    figure.savefig(plot_path / "flip_rates.pdf", bbox_inches="tight")
    figure.savefig(plot_path / "flip_rates.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def analyze(root, precision):
    signature, rows = get_manifest("heldout", root)
    with open_results(root) as db:
        ledger = require_complete(db, "heldout", precision, signature, rows)
        dev_signature, dev_rows = get_manifest("development", root)
        pilot = require_complete(db, "development", precision, dev_signature, dev_rows)
    assert sum(map(len, ledger.values())) == 6000
    assert sum(map(len, pilot.values())) == 1200
    verdicts = {}
    for model, records in ledger.items():
        table = {row_key(record): record["verdict"] for record in records}
        for prompt, variant in product(PROMPTS, (*STYLES, "repeat")):
            verdicts[model, prompt, variant] = np.asarray([
                [table[row["pair_id"], variant, order, prompt] for order in ORDERS] for row in rows
            ])
    metrics, transitions = collect_statistics(rows, verdicts)
    timing = collect_timing(root, ledger, precision)
    chosen = examples(rows, verdicts)
    output = root / "analysis"
    write_csv(root / "heldout_judgments.csv", [dict(row) for values in ledger.values() for row in values])
    write_csv(root / "development_judgments.csv", [dict(row) for values in pilot.values() for row in values])
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "transitions.csv", transitions)
    write_csv(output / "latency_and_memory.csv", timing)
    write_json(output / "examples.json", chosen)
    budget = read_json(root / "runs" / "gpu_budget.json")
    score_ties = sum(
        scores.count(max(scores)) > 1
        for records in ledger.values()
        for scores in map(verdict_logits, records)
    )
    summary = {
        "precision": precision, "heldout_judgments": sum(map(len, ledger.values())),
        "development_judgments": sum(map(len, pilot.values())),
        "gpu_hours_charged": sum(entry["elapsed_seconds"] for entry in budget) / 3600,
        "heldout_hours_charged": sum(entry["elapsed_seconds"] for entry in budget
                                    if entry["split"] == "heldout") / 3600,
        "heldout_equal_top_logits": score_ties,
        "selection": read_json(root / "data" / "selection.json"),
        "forecast": read_json(root / "runs" / f"forecast-{precision}.json"),
    }
    write_json(output / "summary.json", summary)
    plot_results(root, metrics)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("selection", "forecast")}, indent=2))
