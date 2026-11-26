"""Calculate question-clustered correctness and paired style effects."""

from collections import Counter
from itertools import product
import json
import re

import numpy as np

from style_judges.analysis import cluster_draws, write_csv
from style_judges.extension import TOTAL_SECONDS, charged
from style_judges.extension_data import EDIT_STYLES, GOLD_ARMS
from style_judges.inference import (
    get_manifest, open_results, require_complete, row_key, verdict_logits,
)
from style_judges.protocol import ORDERS, PROMPTS, STYLES, read_json, write_json
from style_judges.sources import MODELS


RULES = ("ordered", "pooled", "ordered_forced", "pooled_forced")
IDENTITIES = {"fixed": 0, "target": 1, "tie": 2}
EXPLICIT_FORMAT = re.compile(
    r"\b(?:put|give|write|return|output|format|provide|present|express|state|report|"
    r"answer\s+(?:must|should))\b[^\r\n.?!]{0,80}"
    r"\b(?:bold|boxed|markdown|json|single (?:integer|number|letter)|"
    r"answer string|(?:four|five) times)\b",
    re.IGNORECASE,
)


def decisions(logits, original_verdicts):
    """Project ordered A/B logits into answer identities before probability pooling."""
    logits = np.asarray(logits, dtype=float)
    aligned = logits.copy()
    aligned[:, 1, :2] = logits[:, 1, [1, 0]]
    ordered_forced = logits[:, :, :2].argmax(axis=2)
    ordered_forced[:, 1] = 1 - ordered_forced[:, 1]

    def pool(values):
        exp = np.exp(values - values.max(axis=2, keepdims=True))
        probabilities = exp / exp.sum(axis=2, keepdims=True)
        return probabilities.mean(axis=1).argmax(axis=1)[:, None]

    return dict(zip(RULES, (
        original_verdicts, pool(aligned), ordered_forced, pool(aligned[:, :, :2]),
    ), strict=True))


def get_decisions(rows, records, arms):
    configs = {}
    for model, entries in records.items():
        lookup = {row_key(raw): raw for raw in entries}
        for p, arm in product(PROMPTS, arms):
            matrix = [[lookup[row["pair_id"], arm, order, p] for order in ORDERS] for row in rows]
            scores = [[verdict_logits(value) for value in orders] for orders in matrix]
            ordered = np.array([[IDENTITIES[value["verdict"]] for value in orders] for orders in matrix])
            for rule, outcome in decisions(scores, ordered).items():
                configs[model, p, rule, arm] = outcome
    return configs


class QuestionBootstrap:
    """Equal weights for questions and equal weights within each question."""

    def __init__(self, rows, cohort, precision):
        domains = {row["question_group"]: row["domain"] for row in rows}
        self.groups = sorted(domains)
        group_index = {key: index for index, key in enumerate(self.groups)}
        self.inverse = np.array([group_index[row["question_group"]] for row in rows])
        self.counts = np.bincount(self.inverse)
        self.draws = cluster_draws(np.array([domains[group] for group in self.groups]))
        self.cohort = cohort
        self.precision = precision
        self.metrics = []

    def group_mean(self, values):
        values = np.asarray(values)
        per_pair = values.reshape(len(self.inverse), -1).mean(axis=1)
        return np.bincount(self.inverse, weights=per_pair) / self.counts

    def add(self, metric, model, prompt, rule, condition, values, eligible=None):
        values = np.asarray(values)
        group_values = self.group_mean(values)
        if eligible is None:
            estimate = group_values.mean()
            sampled = group_values[self.draws].mean(axis=1)
            denominator = values.size
        else:
            group_eligible = self.group_mean(eligible)
            denominator = int(np.sum(eligible))
            estimate = group_values.sum() / group_eligible.sum() if denominator else None
            boot_eligible = group_eligible[self.draws].sum(axis=1)
            valid = boot_eligible > 0
            sampled = group_values[self.draws[valid]].sum(axis=1) / boot_eligible[valid]
        low, high = map(float, np.percentile(sampled, [2.5, 97.5])) if len(sampled) else (None, None)
        self.metrics.append({
            "cohort": self.cohort, "precision": self.precision, "model": model, "prompt": prompt,
            "rule": rule, "condition": condition, "metric": metric,
            "estimate": float(estimate) if estimate is not None else None,
            "ci_low": low, "ci_high": high, "question_groups": len(self.groups),
            "pairs": len(self.inverse), "raw_numerator": int(values.sum()), "raw_denominator": denominator,
        })


def labeled_statistics(rows, configs, precision):
    stats = QuestionBootstrap(rows, "judgebench", precision)
    gold = np.array([IDENTITIES[row["reference"]] for row in rows])[:, None]
    for model, prompt, rule in product(MODELS, PROMPTS, RULES):
        baseline = configs[model, prompt, rule, "original"]
        initial_correct = baseline == gold
        for arm in GOLD_ARMS:
            output = configs[model, prompt, rule, arm]
            chosen = output != IDENTITIES["tie"]
            is_correct = output == gold
            wrong = chosen & ~is_correct
            stats.add("accuracy", model, prompt, rule, arm, is_correct)
            stats.add("coverage", model, prompt, rule, arm, chosen)
            stats.add("conditional_accuracy", model, prompt, rule, arm, is_correct, chosen)
            if arm == "original":
                continue
            stats.add("correctness_loss", model, prompt, rule, arm,
                      initial_correct.astype(int) - is_correct)
            stats.add("preference_flip", model, prompt, rule, arm, baseline != output)
            if arm.endswith("_incorrect"):
                stats.add("correct_to_wrong", model, prompt, rule, arm, initial_correct & wrong, initial_correct)
                stats.add("correct_to_tie", model, prompt, rule, arm, initial_correct & ~chosen, initial_correct)
        for style in EDIT_STYLES:
            on_correct = configs[model, prompt, rule, f"{style}_correct"] == gold
            on_incorrect = configs[model, prompt, rule, f"{style}_incorrect"] == gold
            stats.add("mirrored_accuracy_advantage", model, prompt, rule, style,
                      on_correct.astype(int) - on_incorrect)
    losses = {}
    for model, prompt, rule in product(MODELS, PROMPTS, RULES):
        base = (configs[model, prompt, rule, "original"] == gold).astype(int)
        edited = configs[model, prompt, rule, "verbosity_incorrect"] == gold
        losses[model, prompt, rule] = base - edited
    for model, rule in product(MODELS, RULES):
        stats.add("rubric_reduction_in_loss", model, "brief_minus_rubric", rule, "verbosity_incorrect",
                  losses[model, "brief", rule] - losses[model, "rubric", rule])
    for model, prompt, pair in product(MODELS, PROMPTS, (("ordered", "pooled"),
                                                        ("ordered_forced", "pooled_forced"))):
        before, after = pair
        stats.add("pooling_reduction_in_loss", model, prompt, after, "verbosity_incorrect",
                  losses[model, prompt, before].mean(axis=1) - losses[model, prompt, after].mean(axis=1))
    return stats.metrics


def calm_statistics(rows, configs):
    stats = QuestionBootstrap(rows, "calm", "nf4")
    for model, prompt, arm in product(MODELS, PROMPTS, STYLES):
        baseline = configs[model, prompt, "ordered", "original"]
        predicted = configs[model, prompt, "ordered", arm]
        stats.add("order_consistency", model, prompt, "ordered", arm, predicted[:, 0] == predicted[:, 1])
        stats.add("coverage", model, prompt, "ordered", arm, predicted != IDENTITIES["tie"])
        if arm != "original":
            stats.add("preference_flip", model, prompt, "ordered", arm, predicted != baseline)
    return stats.metrics


def precision_statistics(rows, nf4, bf16):
    stats = QuestionBootstrap(rows, "judgebench_matched", "bf16_minus_nf4")
    gold = np.array([IDENTITIES[row["reference"]] for row in rows])[:, None]
    for model, prompt, rule, arm in product(MODELS, PROMPTS, RULES, GOLD_ARMS):
        old = nf4[model, prompt, rule, arm]
        new = bf16[model, prompt, rule, arm]
        stats.add("accuracy_delta", model, prompt, rule, arm, (new == gold).astype(int) - (old == gold))
        stats.add("coverage_delta", model, prompt, rule, arm,
                  (new != IDENTITIES["tie"]).astype(int) - (old != IDENTITIES["tie"]))
    return stats.metrics


def format_sensitivity(rows, configs):
    # This sensitivity check is exploratory. Keep every pair in the frozen primary analysis.
    keep = np.array([not EXPLICIT_FORMAT.search(row["question"]) for row in rows])
    subset = [row for row, retained in zip(rows, keep, strict=True) if retained]
    if not subset:
        return [], 0
    stats = QuestionBootstrap(subset, "judgebench_format_screen", "nf4")
    gold = np.array([IDENTITIES[row["reference"]] for row in subset])[:, None]
    model = "Qwen3-4B"
    for prompt, rule in product(PROMPTS, ("ordered", "pooled")):
        base = configs[model, prompt, rule, "original"][keep] == gold
        for arm in GOLD_ARMS[1:]:
            edited = configs[model, prompt, rule, arm][keep] == gold
            stats.add("correctness_loss", model, prompt, rule, arm,
                      base.astype(int) - edited)
    return stats.metrics, len(subset)


def case_examples(rows, configs):
    selected = []
    seen = set()
    for style in EDIT_STYLES:
        ranked = []
        for i, row in enumerate(rows):
            if row["question_group"] in seen:
                continue
            correct = IDENTITIES[row["reference"]]
            mistakes = []
            for model, prompt in product(MODELS, PROMPTS):
                old = configs[model, prompt, "ordered", "original"][i]
                after = configs[model, prompt, "ordered", f"{style}_incorrect"][i]
                mistakes.extend(
                    {"model": model, "prompt": prompt, "order": order}
                    for order, a, b in zip(ORDERS, old, after, strict=True)
                    if a == correct and b == 1 - correct
                )
            if mistakes:
                key = (-len({item["model"] for item in mistakes}), -len(mistakes), row["pair_id"])
                ranked.append((key, i, mistakes))
        if ranked:
            _, index, mistakes = min(ranked)
            row = rows[index]
            selected.append({"style": style, "row": row, "correct_to_wrong": mistakes})
            seen.add(row["question_group"])
    return selected


def plot_changes(root, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["model"], row["condition"]): row for row in metrics
              if row["cohort"] == "judgebench" and row["precision"] == "nf4"
              and row["prompt"] == "brief" and row["rule"] == "ordered"
              and row["metric"] == "correctness_loss"}
    plt.rcParams.update({
        "font.size": 9, "axes.spines.right": False, "axes.spines.top": False,
        "axes.labelcolor": "#202b37", "text.color": "#17212a", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharex=True, sharey=True, layout="constrained")
    for axis, style in zip(axes, EDIT_STYLES, strict=True):
        axis.axvline(0, color="#647180", lw=.7)
        for pos, model in enumerate(MODELS):
            for side, color, marker, shift in (
                ("correct", "#117882", "o", -.13), ("incorrect", "#ae462a", "s", .13),
            ):
                row = lookup[model, f"{style}_{side}"]
                value = -100 * row["estimate"]
                low, high = -100 * row["ci_high"], -100 * row["ci_low"]
                axis.errorbar(value, pos + shift, xerr=[[max(0, value - low)], [max(0, high - value)]],
                              fmt=marker, color=color, markersize=4.5, capsize=2.5, elinewidth=1,
                              label=f"Edited {side} answer" if pos == 0 else None)
        axis.grid(axis="x", alpha=.22)
        axis.set_title(style.capitalize(), fontweight="bold")
        axis.set_yticks(range(len(MODELS)), [model.removeprefix("Qwen3-") for model in MODELS])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Qwen3 checkpoint")
    axes[1].set_xlabel("Change in judge accuracy (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, fontsize=8)
    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "accuracy_changes.pdf", bbox_inches="tight")
    fig.savefig(folder / "accuracy_changes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_raw(root, name, rows, records):
    metadata = {row["pair_id"]: row for row in rows}
    data = []
    for entries in records.values():
        for raw in entries:
            row = metadata[raw["pair_id"]]
            data.append({
                **dict(raw), "cohort": row["cohort"], "domain": row["domain"],
                "question_group": row["question_group"], "generator": row.get("generator", ""),
                "reference": row.get("reference", ""),
            })
    write_csv(root / f"{name}_judgments.csv", data)


def verify_precision_inputs(nf4, bf16):
    for model, entries in bf16.items():
        fingerprints = {row_key(record): record["input_hash"] for record in nf4[model]}
        if any(fingerprints.get(row_key(record)) != record["input_hash"] for record in entries):
            raise RuntimeError("The BF16 comparison changed a matched input")


def analyze_extension(root):
    plan = read_json(root / "runs" / "plan.json")
    core_sha, core = get_manifest("core", root)
    pilot_sha, pilot = get_manifest("pilot", root)
    if core_sha != plan["core_manifest_sha256"]:
        raise RuntimeError("Unexpected held-out manifest")
    assert {row["question_group"] for row in pilot}.isdisjoint(row["question_group"] for row in core)
    with open_results(root) as db:
        primary = require_complete(db, "core", "nf4", core_sha, core)
        development = require_complete(db, "pilot", "nf4", pilot_sha, pilot)
        if plan["precision_completed"]:
            precision_sha, subset = get_manifest("precision", root)
            precision = require_complete(db, "precision", "bf16", precision_sha, subset)
        else:
            subset = []
    judge = [row for row in core if row["cohort"] == "judgebench"]
    calm = [row for row in core if row["cohort"] == "calm"]
    configs = get_decisions(judge, primary, GOLD_ARMS)
    metrics = labeled_statistics(judge, configs, "nf4")
    screened_metrics, screened_pairs = format_sensitivity(judge, configs)
    metrics += screened_metrics
    metrics += calm_statistics(calm, get_decisions(calm, primary, STYLES))
    export_raw(root, "core_nf4", core, primary)
    export_raw(root, "pilot_nf4", pilot, development)
    if subset:
        verify_precision_inputs(primary, precision)
        bf16 = get_decisions(subset, precision, GOLD_ARMS)
        nf4 = get_decisions(subset, primary, GOLD_ARMS)
        metrics += labeled_statistics(subset, bf16, "bf16")
        matched = labeled_statistics(subset, nf4, "nf4_matched")
        metrics += matched + precision_statistics(subset, nf4, bf16)
        export_raw(root, "precision_bf16", subset, precision)
    output = root / "analysis"
    write_csv(output / "metrics.csv", metrics)
    write_json(output / "examples.json", case_examples(judge, configs))
    ledger = read_json(root / "runs" / "gpu_budget.json")
    total_seconds = charged(ledger) + plan["original_seconds"]
    assert total_seconds <= TOTAL_SECONDS
    summary = {
        "original_gpu_hours": plan["original_seconds"] / 3600,
        "extension_gpu_hours": charged(ledger) / 3600, "total_gpu_hours": total_seconds / 3600,
        "gpu_hours_by_stage": {stage: charged(ledger, stage) / 3600 for stage in ("pilot", "core", "precision")},
        "pilot_calls": sum(map(len, development.values())), "core_calls": sum(map(len, primary.values())),
        "precision_calls": sum(map(len, precision.values())) if subset else 0,
        "judgebench_pairs": len(judge), "judgebench_questions": len({row["question_group"] for row in judge}),
        "judgebench_domains": dict(Counter(row["domain"] for row in judge)),
        "judgebench_generators": dict(Counter(row["generator"] for row in judge)),
        "calm_pairs": len(calm), "calm_sources": dict(Counter(row["source"] for row in calm)),
        "precision_pairs": len(subset), "core_manifest_sha256": core_sha,
        "format_sensitivity_pairs": screened_pairs,
        "precision_started": plan["precision_started"], "core_trimmed": plan["core_trimmed"],
        "forecast_gpu_hours": plan["core_forecast"]["total_seconds"] / 3600,
        "primary_result": next(row for row in metrics if row["cohort"] == "judgebench"
                               and row["precision"] == "nf4" and row["model"] == "Qwen3-4B"
                               and row["prompt"] == "brief" and row["rule"] == "ordered"
                               and row["condition"] == "verbosity_incorrect"
                               and row["metric"] == "correctness_loss"),
    }
    write_json(output / "summary.json", summary)
    plot_changes(root, metrics)
    print(json.dumps(summary, indent=2), flush=True)
