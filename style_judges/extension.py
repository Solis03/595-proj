"""Run the accuracy and style extension under a shared six-hour GPU budget."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import subprocess

from style_judges.extension_data import (
    PRECISION_GROUPS, data_manifest, fetch_judgebench, prepare_extension, ranked_groups,
)
from style_judges.inference import (
    case_keys, exclusive_supervisor, get_manifest, launch_worker, open_results, require_complete,
    row_key, rows_for,
)
from style_judges.protocol import RESULTS, manifest_path, read_json, write_json
from style_judges.sources import MODELS


TOTAL_SECONDS = 6 * 3600
PILOT_SECONDS = .35 * 3600
CORE_ALLOCATION = 3 * 3600
CORE_WITH_RESERVE = 4.5 * 3600
PRECISION_SECONDS = 3600
LENGTH_BANDS = (512, 1024, 2048, 3072, 4096)
EXTENSION_RESULTS = RESULTS / "extension"


def original_seconds():
    path = RESULTS / "runs" / "gpu_budget.json"
    if path.exists():
        entries = read_json(path)
        if any(entry["status"] == "running" for entry in entries):
            raise RuntimeError("The original GPU ledger has an active worker")
        return sum(entry["elapsed_seconds"] for entry in entries)
    return read_json(RESULTS / "analysis" / "summary.json")["gpu_hours_charged"] * 3600


def charged(entries, stage=None):
    return sum(entry["elapsed_seconds"] for entry in entries
               if stage is None or entry["split"] == stage)


def run_stage(root, split, precision, ceiling, prior, path, ledger):
    signature, rows = get_manifest(split, root)
    expected = case_keys(rows)
    with open_results(root) as db:
        for model in MODELS:
            observed = {row_key(row) for row in rows_for(db, split, precision, signature, model)}
            if observed == expected:
                print(f"{split} | {precision} | {model} already complete", flush=True)
                continue
            if observed - expected:
                raise RuntimeError(f"{model} has judgments outside the frozen {split} matrix")
            left = min(ceiling - charged(ledger, split), TOTAL_SECONDS - prior - charged(ledger))
            launch_worker(root, split, precision, model, left, path, ledger)
        records = require_complete(db, split, precision, signature, rows)
    print(f"Verified {sum(map(len, records.values())):,} {split} {precision} judgments", flush=True)
    return records


def length_band(tokens):
    return next(band for band in LENGTH_BANDS if tokens <= band)


def timing_profile(root, pilot, ledger):
    profile = {}
    for model, records in pilot.items():
        timings = defaultdict(list)
        for item in records:
            timings[length_band(item["input_tokens"])].append(item["latency_ms"] / 1000)
        metadata = read_json(root / "runs" / f"pilot-nf4-{model}.json")
        load = metadata["load_seconds_latest"] + 12
        elapsed = sum(entry["elapsed_seconds"] for entry in ledger
                      if entry["split"] == "pilot" and entry["model"] == model)
        gpu_sum = sum(row["latency_ms"] for row in records) / 1000
        overhead = max(.008, (elapsed - gpu_sum - load) / len(records))
        high = 0.0
        last_band = min(timings)
        bands = {}
        for band in LENGTH_BANDS:
            samples = timings[band]
            if len(samples) >= 2:
                estimate = statistics.quantiles(samples, n=20, method="inclusive")[18]
                last_band = band
                high = max(high, estimate)
                seconds = high
            elif samples:
                seconds = max(high, samples[0])
            elif high > 0:
                seconds = high * band / last_band
            else:
                seconds = min(value for sample in timings.values() for value in sample)
            bands[str(band)] = {
                "samples": len(samples), "seconds_per_call": 1.25 * seconds + overhead,
            }
        profile[model] = {"startup_seconds": load, "bands": bands}
    return profile


def forecast(rows, profile):
    histogram = Counter(length_band(length) for row in rows for length in row["case_tokens"].values())
    by_model = {
        model: spec["startup_seconds"] + sum(
            count * spec["bands"][str(band)]["seconds_per_call"] for band, count in histogram.items()
        )
        for model, spec in profile.items()
    }
    return {"by_model": by_model, "total_seconds": sum(by_model.values())}


def choose_core(candidates, profile, available):
    judge = [row for row in candidates if row["cohort"] == "judgebench"]
    calm = [row for row in candidates if row["cohort"] == "calm"]
    full = forecast(candidates, profile)
    core_limit = min(CORE_WITH_RESERVE, available)
    if full["total_seconds"] <= core_limit:
        return candidates, full, False
    calm_ids = set(ranked_groups(calm, "trim-calm", 100))
    calm = [row for row in calm if row["question_group"] in calm_ids]
    ranking = ranked_groups(judge, "trim-judgebench", len(judge))
    groups = defaultdict(list)
    for row in judge:
        groups[row["question_group"]].append(row)
    rows = calm + judge
    # Remove only complete groups from the end of the frozen stratified order.
    for group in reversed(ranking):
        estimate = forecast(rows, profile)
        if estimate["total_seconds"] <= core_limit:
            if {row["domain"] for row in rows if row["cohort"] == "judgebench"} != {
                "math", "reasoning", "knowledge"
            }:
                break
            return rows, estimate, True
        members = {row["pair_id"] for row in groups[group]}
        rows = [row for row in rows if row["pair_id"] not in members]
    raise RuntimeError("The pilot forecast cannot fit a complete held-out comparison within the cap")


def precision_rows(core):
    judge = sorted((row for row in core if row["cohort"] == "judgebench"),
                   key=lambda row: row["pair_id"])
    selection = set(ranked_groups(judge, "precision", PRECISION_GROUPS))
    representative = {}
    for row in judge:
        if row["question_group"] in selection:
            representative.setdefault(row["question_group"], row)
    return list(representative.values())


def supervise_extension(root):
    # Share the original local supervisor's lock across both suites.
    with exclusive_supervisor(RESULTS), exclusive_supervisor(root):
        prior = original_seconds()
        ledger_path = root / "runs" / "gpu_budget.json"
        ledger = read_json(ledger_path) if ledger_path.exists() else []
        if any(entry["status"] == "running" for entry in ledger):
            raise RuntimeError("Reconcile the interrupted GPU worker and its elapsed time first")
        pilot = run_stage(root, "pilot", "nf4", PILOT_SECONDS, prior, ledger_path, ledger)
        profile = timing_profile(root, pilot, ledger)
        plan_path = root / "runs" / "plan.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            signature, core = get_manifest("core", root)
            if signature != plan["core_manifest_sha256"]:
                raise RuntimeError("The core manifest no longer matches the frozen run plan")
        else:
            candidates = read_json(root / "data" / "candidates.json")
            available = TOTAL_SECONDS - prior - charged(ledger)
            core, prediction, trimmed = choose_core(candidates, profile, available - 5)
            signature = data_manifest(root, "core", core)
            judge = [row for row in core if row["cohort"] == "judgebench"]
            plan = {
                "cap_seconds": TOTAL_SECONDS, "original_seconds": prior, "pilot_profile": profile,
                "core_manifest_sha256": signature, "core_forecast": prediction,
                "core_calls": len(case_keys(core)) * len(MODELS), "core_trimmed": trimmed,
                "judgebench_pairs": len(judge),
                "judgebench_questions": len({row["question_group"] for row in judge}),
                "calm_pairs": len(core) - len(judge),
                "precision_planned": not trimmed and prediction["total_seconds"] <= CORE_ALLOCATION,
            }
            write_json(plan_path, plan)
        print("Frozen core:", json.dumps({key: plan[key] for key in (
            "judgebench_pairs", "judgebench_questions", "calm_pairs", "core_calls", "core_forecast",
        )}), flush=True)
        run_stage(root, "core", "nf4", CORE_WITH_RESERVE, prior, ledger_path, ledger)

        subset = precision_rows(core)
        estimate = forecast(subset, profile)["total_seconds"] * 1.4
        available = TOTAL_SECONDS - prior - charged(ledger)
        if "precision_started" not in plan:
            plan["precision_started"] = (
                plan["precision_planned"] and len(subset) == PRECISION_GROUPS
                and estimate <= min(PRECISION_SECONDS, available - 5)
            )
            plan["precision_forecast_seconds"] = estimate
            write_json(plan_path, plan)
        if plan["precision_started"]:
            data_manifest(root, "precision", subset)
            try:
                run_stage(root, "precision", "bf16", PRECISION_SECONDS, prior, ledger_path, ledger)
                plan["precision_completed"] = True
            except (TimeoutError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                plan["precision_completed"] = False
                plan["precision_issue"] = str(error)
                print(f"BF16 follow-up incomplete: {error}", flush=True)
        else:
            plan["precision_completed"] = False
        write_json(plan_path, plan)
        print(f"Cumulative GPU worker time: {(prior + charged(ledger)) / 3600:.3f} h", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=EXTENSION_RESULTS)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "fetch":
        fetch_judgebench()
    elif args.command == "prepare":
        prepare_extension(root)
    elif args.command == "run":
        supervise_extension(root)
    else:
        from style_judges.extension_analysis import analyze_extension

        analyze_extension(root)


if __name__ == "__main__":
    main()
