"""Local Qwen inference, durable per-judgment results, and GPU time limits."""

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
import json
import math
import os
import random
import sqlite3
import statistics
import struct
import subprocess
import sys
import time

from style_judges.protocol import (
    LABELS, MAX_TOKENS, ORDERS, PROMPTS, SEED, VARIANTS, canonical_json, label_ids,
    load_tokenizer, manifest_path, read_json, render, sha256, token_ids, verdict_identity,
    write_json,
)
from style_judges.sources import ASSETS, MODELS, ROOT


STAGE_LIMITS = {"development": 2 * 3600, "heldout": 18 * 3600}
RECOVERY_LIMIT = 4 * 3600
TOTAL_LIMIT = 24 * 3600
SCHEMA = """
CREATE TABLE IF NOT EXISTS judgments (
    split TEXT NOT NULL,
    precision TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    revision TEXT NOT NULL,
    pair_id TEXT NOT NULL,
    source TEXT NOT NULL,
    variant TEXT NOT NULL,
    answer_order TEXT NOT NULL,
    prompt TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    label TEXT NOT NULL CHECK(label IN ('A', 'B', 'T')),
    verdict TEXT NOT NULL CHECK(verdict IN ('fixed', 'target', 'tie')),
    logit_a REAL NOT NULL,
    logit_b REAL NOT NULL,
    logit_t REAL NOT NULL,
    latency_ms REAL NOT NULL,
    PRIMARY KEY (split, precision, manifest_hash, model, pair_id, variant, answer_order, prompt)
) WITHOUT ROWID;
"""


def open_results(root):
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "judgments.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def get_manifest(split, root):
    manifest = read_json(manifest_path(split, root))
    signature = manifest.pop("manifest_sha256")
    assert sha256(canonical_json(manifest).encode("utf-8")) == signature
    assert manifest["model_revisions"] == MODELS and manifest["prompts"] == PROMPTS
    assert manifest["split"] == split and manifest["max_input_tokens"] == MAX_TOKENS
    return signature, manifest["rows"]


def case_keys(rows):
    return set(product((row["pair_id"] for row in rows), VARIANTS, ORDERS, PROMPTS))


def rows_for(db, split, precision, signature, model):
    return db.execute(
        "SELECT * FROM judgments WHERE split=? AND precision=? AND manifest_hash=? AND model=? "
        "ORDER BY pair_id, variant, answer_order, prompt",
        (split, precision, signature, model),
    ).fetchall()


def row_key(row):
    return row["pair_id"], row["variant"], row["answer_order"], row["prompt"]


def verdict_logits(row):
    return [row[f"logit_{label.lower()}"] for label in LABELS]


def require_complete(db, split, precision, signature, rows, models=MODELS):
    expected = case_keys(rows)
    observed_by_model = {}
    for model in models:
        records = rows_for(db, split, precision, signature, model)
        observed = {row_key(record) for record in records}
        if observed != expected:
            raise RuntimeError(f"{split} {model} {precision}: {len(observed)} of {len(expected)} rows")
        by_key = {row_key(record): record for record in records}
        for raw in records:
            assert raw["revision"] == MODELS[model] and 0 < raw["input_tokens"] <= MAX_TOKENS
            assert raw["verdict"] == verdict_identity(raw["label"], raw["answer_order"])
            scores = verdict_logits(raw)
            assert raw["label"] == LABELS[max(range(len(LABELS)), key=scores.__getitem__)]
            if raw["variant"] == "repeat":
                orig_key = (raw["pair_id"], "original", raw["answer_order"], raw["prompt"])
                assert by_key[orig_key]["input_hash"] == raw["input_hash"]
                assert by_key[orig_key]["input_tokens"] == raw["input_tokens"]
        observed_by_model[model] = records
    return observed_by_model


def input_case(tokenizer, row, style, order, prompt):
    text = render(tokenizer, row, style, order, prompt)
    if not text.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("Qwen non-thinking chat template missing closed think block")
    ids = token_ids(tokenizer, text)
    if not 0 < len(ids) <= MAX_TOKENS:
        raise ValueError(f"{row['pair_id']} input contains {len(ids)} tokens")
    signature = sha256(struct.pack(f"<{len(ids)}I", *ids))
    return ids, signature


def worker(root, split, precision, model_name):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    device = torch.device("cuda:0")
    signature, rows = get_manifest(split, root)
    _, dev_rows = get_manifest("development", root)
    tokenizer = load_tokenizer(model_name)
    choices = torch.tensor(label_ids(tokenizer), device=device)
    assert (torch.arange(4, device=device, dtype=torch.bfloat16) + 1).cpu().tolist() == [1, 2, 3, 4]
    db = open_results(root)
    done = {row_key(record) for record in rows_for(db, split, precision, signature, model_name)}
    remaining = case_keys(rows) - done
    if not remaining:
        return

    label_probe = render(tokenizer, dev_rows[0], "original", "fixed_first", "brief")
    for label, token in zip(LABELS, choices.tolist(), strict=True):
        assert token_ids(tokenizer, label_probe + label) == token_ids(tokenizer, label_probe) + [token]

    metadata_path = root / "runs" / f"{split}-{precision}-{model_name}.json"
    print(f"{split} | {model_name} {precision} | {len(done)} saved, {len(remaining)} to run", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    load_start = time.monotonic()
    options = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if precision == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    model = AutoModelForCausalLM.from_pretrained(
        ASSETS / "models" / model_name, **options, use_safetensors=True,
        local_files_only=True, trust_remote_code=False,
    ).eval()
    torch.cuda.synchronize(device)
    load_s = time.monotonic() - load_start
    load_allocated = torch.cuda.max_memory_allocated(device)
    load_reserved = torch.cuda.max_memory_reserved(device)
    if precision == "nf4" and not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Requested NF4, but model did not load in 4-bit mode")

    def evaluate(ids):
        batch = torch.tensor([ids], device=device, dtype=torch.long)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(input_ids=batch, logits_to_keep=1, use_cache=False)
            scores = output.logits[0, -1].index_select(0, choices).float().cpu().tolist()
        delay_ms = (time.perf_counter() - started) * 1000
        if not all(math.isfinite(score) for score in scores):
            raise FloatingPointError(f"Non-finite verdict scores: {scores}")
        return scores, delay_ms

    torch.cuda.reset_peak_memory_stats(device)
    warm_cases = sorted(dev_rows, key=lambda row: row["max_prompt_tokens"])
    warm_examples = (warm_cases[0], warm_cases[len(warm_cases) // 2], warm_cases[-1])
    for example, variant in product(warm_examples, ("original", "verbosity")):
        ids, _ = input_case(tokenizer, example, variant, "fixed_first", "rubric")
        evaluate(ids)

    def update_metadata(completed):
        previous = read_json(metadata_path) if metadata_path.is_file() else {}
        telemetry = {
            "model": model_name, "precision": precision, "manifest_sha256": signature,
            "revision": MODELS[model_name], "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "warmup_calls_per_session": 6,
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "load_seconds_latest": load_s,
            "completed": completed,
        }
        if precision == "nf4":
            import bitsandbytes

            telemetry["bitsandbytes"] = bitsandbytes.__version__
        metrics = {
            "load_peak_allocated_bytes": load_allocated,
            "load_peak_reserved_bytes": load_reserved,
            "inference_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "inference_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        for key, value in metrics.items():
            telemetry[key] = max(value, previous.get(key, 0)) if previous.get("manifest_sha256") == signature else value
        write_json(metadata_path, telemetry)

    by_id = {row["pair_id"]: row for row in rows}
    jobs = []
    rng = random.Random(SEED)
    for is_repeat in (False, True):
        wave = sorted(key for key in remaining if (key[1] == "repeat") == is_repeat)
        rng.shuffle(wave)
        jobs.extend(wave)

    count = len(done)
    update_metadata(count)
    for pair_id, variant, order, prompt in jobs:
        row = by_id[pair_id]
        ids, fingerprint = input_case(tokenizer, row, variant, order, prompt)
        scores, latency = evaluate(ids)
        label = LABELS[max(range(len(LABELS)), key=scores.__getitem__)]
        with db:
            db.execute(
                "INSERT INTO judgments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    split, precision, signature, model_name, MODELS[model_name], pair_id, row["source"],
                    variant, order, prompt, fingerprint, len(ids), label, verdict_identity(label, order),
                    *scores, latency,
                ),
            )
        count += 1
        if count % 100 == 0 or count == len(done) + len(jobs):
            update_metadata(count)
            print(f"{split} | {model_name} | {count}/{len(done) + len(jobs)}", flush=True)
    db.close()


@contextmanager
def exclusive_supervisor(root):
    # A separate SQLite lock serializes GPU jobs without blocking judgment writes.
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "supervisor.sqlite", timeout=1)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        yield
    finally:
        connection.close()


def ledger_remaining(entries, split):
    used = defaultdict(float)
    for entry in entries:
        used[entry["split"]] += entry["elapsed_seconds"]
    total = sum(used.values())
    stage_left = STAGE_LIMITS[split] - used[split]
    spent_recovery = sum(max(0, seconds - STAGE_LIMITS[stage]) for stage, seconds in used.items())
    recovery_left = RECOVERY_LIMIT - spent_recovery
    return min(TOTAL_LIMIT - total, max(0, stage_left) + max(0, recovery_left))


def estimate_heldout(root, precision):
    signature, rows = get_manifest("development", root)
    _, heldout = get_manifest("heldout", root)
    with open_results(root) as db:
        pilot = require_complete(db, "development", precision, signature, rows)
    forecast = {}
    for model, records in pilot.items():
        p95 = statistics.quantiles([row["latency_ms"] for row in records], n=20, method="inclusive")[18]
        meta = read_json(root / "runs" / f"development-{precision}-{model}.json")
        # Wall-time allowance includes tokenization, disk commits, and Windows scheduling.
        seconds = meta["load_seconds_latest"] + len(case_keys(heldout)) * (3 * p95 / 1000 + 0.05)
        forecast[model] = {"pilot_p95_ms": p95, "forecast_wall_seconds": seconds}
    result = {"precision": precision, "by_model": forecast,
              "total_seconds": sum(value["forecast_wall_seconds"] for value in forecast.values())}
    write_json(root / "runs" / f"forecast-{precision}.json", result)
    print("Pilot-based held-out forecast:", json.dumps(result), flush=True)
    return result


def supervise(root, split, precision, model=None):
    if split == "heldout" and model is not None:
        raise ValueError("The held-out stage requires the complete model matrix")
    active_models = (model,) if model else MODELS
    with exclusive_supervisor(root):
        signature, rows = get_manifest(split, root)
        path = root / "runs" / "gpu_budget.json"
        ledger = read_json(path) if path.is_file() else []
        if any(entry["status"] == "running" for entry in ledger):
            raise RuntimeError("A prior GPU worker was interrupted unexpectedly. Reconcile its process and elapsed time.")
        if split == "heldout":
            forecast = estimate_heldout(root, precision)
            # Full-run forecast is only needed before the first held-out launch.
            new_run = not any(entry["split"] == split and entry["precision"] == precision for entry in ledger)
            if new_run and forecast["total_seconds"] > min(STAGE_LIMITS[split], ledger_remaining(ledger, split)):
                raise RuntimeError("Measured pilot throughput exceeds the held-out budget")
        expected = case_keys(rows)
        with open_results(root) as db:
            for active_model in active_models:
                existing = rows_for(db, split, precision, signature, active_model)
                if {row_key(row) for row in existing} == expected:
                    print(f"Already complete | {split} | {precision} | {active_model}", flush=True)
                    continue
                left = ledger_remaining(ledger, split)
                if left < 2:
                    raise TimeoutError("No GPU hours remain")
                entry = {
                    "split": split, "precision": precision, "model": active_model,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                    "elapsed_seconds": 0,
                }
                ledger.append(entry)
                write_json(path, ledger)
                command = [sys.executable, "-m", "style_judges", "_worker", split, precision, active_model,
                           "--output", str(root)]
                started = time.monotonic()
                status = "failed"
                try:
                    environment = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                                       TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                    with subprocess.Popen(command, cwd=ROOT, env=environment) as process:
                        entry["worker_pid"] = process.pid
                        try:
                            write_json(path, ledger)
                            code = process.wait(timeout=max(0.1, left - 1))
                        finally:
                            if process.poll() is None:
                                process.kill()
                                process.wait()
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                    status = "complete"
                finally:
                    entry["elapsed_seconds"] = time.monotonic() - started
                    entry["status"] = status
                    write_json(path, ledger)
            require_complete(db, split, precision, signature, rows, active_models)
        print(f"Verified {len(expected) * len(active_models)} "
              f"{split} judgments in {precision}.", flush=True)
        if split == "development" and model is None:
            estimate_heldout(root, precision)
