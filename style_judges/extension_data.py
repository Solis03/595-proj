"""Prepare the frozen CALM replication and labeled JudgeBench comparisons."""

from collections import Counter, defaultdict
from itertools import product
import json

from style_judges.data import (
    CONTROL_TOKEN, STYLE_REQUEST, VISUAL, apply_edits, candidate_rows, normalized, validate_edits,
)
from style_judges.inference import get_manifest
from style_judges.protocol import (
    EXTENSION_MAX_TOKENS, MAX_TOKENS, ORDERS, PROMPTS, RESULTS, SEED, STYLES,
    canonical_json, load_tokenizer, manifest_path, read_json, render, sha256, write_json,
)
from style_judges.sources import ASSETS, CALM, GSM, MATH, MODELS, SCIENCE, fetch_group


JUDGEBENCH = "e2c52c284e735e139b3daa61c206ee208f36c461"
GENERATORS = ("gpt-4o-2024-05-13", "claude-3-5-sonnet-20240620")
EDIT_STYLES = STYLES[1:]
GOLD_ARMS = (
    "original", *(f"{style}_{side}" for style in EDIT_STYLES for side in ("correct", "incorrect")),
    "verbosity_both",
)
DEV_GROUPS = 40
PRECISION_GROUPS = 90


def judgebench_path(generator):
    return ASSETS / "data" / "judgebench" / f"{generator}.jsonl"


def fetch_judgebench():
    base = f"https://raw.githubusercontent.com/ScalerLab/JudgeBench/{JUDGEBENCH}/data/"
    fetch_group((f"{base}dataset=judgebench,response_model={name}.jsonl", judgebench_path(name))
                for name in GENERATORS)


def digest(value):
    return sha256(canonical_json(value).encode("utf-8"))


def ranked_groups(rows, purpose, limit):
    """A deterministic proportional ordering that also supports nested prefixes."""
    strata = defaultdict(set)
    for row in rows:
        strata[row["domain"]].add(row["question_group"])
    balanced = []
    for domain, group_ids in strata.items():
        ordered = sorted(group_ids, key=lambda group: (digest([SEED, purpose, domain, group]), group))
        balanced.extend(((i + .5) / len(ordered), domain, group) for i, group in enumerate(ordered))
    return [group for _, _, group in sorted(balanced)[:limit]]


def coarse_domain(source):
    if source == "livebench-math":
        return "math"
    if source == "livebench-reasoning":
        return "reasoning"
    if source.startswith("mmlu-pro-"):
        return "knowledge"
    raise ValueError(f"Unexpected JudgeBench source: {source}")


def upstream_question_key(source, original_id, question):
    if original_id is None or original_id == "":
        original_id = "text-" + digest(normalized(question))
    return f"judge:{source}:{original_id}"


def mirrored_arms(a, b, correct):
    edits = [apply_edits(answer, None) for answer in (a, b)]
    if any(edit is None for edit in edits):
        return None
    for variants, evidence in edits:
        validate_edits({"variants": variants, "edit_evidence": evidence})
    edited_a, edited_b = [edit[0] for edit in edits]
    arms = {"original": [a, b]}
    for style in EDIT_STYLES:
        change_a = [edited_a[style], b]
        change_b = [a, edited_b[style]]
        arms[f"{style}_correct"], arms[f"{style}_incorrect"] = (
            (change_a, change_b) if correct == "fixed" else (change_b, change_a)
        )
    arms["verbosity_both"] = [edited_a["verbosity"], edited_b["verbosity"]]
    return arms


def prompt_lengths(row, tokenizers):
    max_lengths = {}
    for tokenizer in tokenizers:
        cases = list(product(row["answer_variants"], ORDERS, PROMPTS))
        texts = [render(tokenizer, row, *case) for case in cases]
        lengths = tokenizer(texts, add_special_tokens=False, return_length=True)["length"]
        for case, length in zip(cases, lengths, strict=True):
            key = "/".join(case)
            max_lengths[key] = max(max_lengths.get(key, 0), length)
    return max_lengths


def annotate_length(row, tokenizers):
    row["case_tokens"] = prompt_lengths(row, tokenizers)
    row["max_prompt_tokens"] = max(row["case_tokens"].values())
    return row["max_prompt_tokens"] <= row["input_limit"]


def group_judgebench(rows):
    # Link on source ID or exact normalized question, including across generators.
    parent = {}
    by_question = {}

    def find(group):
        while parent[group] != group:
            group = parent[group]
        return group

    for row in rows:
        group = row["source_id"]
        parent.setdefault(group, group)
        previous = by_question.setdefault(normalized(row["question"]), group)
        a, b = sorted((find(previous), find(group)))
        parent[b] = a
    for row in rows:
        row["question_group"] = find(row["source_id"])


def judgebench_rows(tokenizers, excluded_questions):
    rows = []
    counts = Counter()
    all_groups = []
    for generator in GENERATORS:
        for line in judgebench_path(generator).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            source = record["source"]
            question, a, b = (record[key] for key in ("question", "response_A", "response_B"))
            if source.startswith("livecodebench"):
                counts["coding"] += 1
                continue
            label = record["label"]
            text = "\n".join((question, a, b))
            source_id = upstream_question_key(source, record["original_id"], question)
            row = {
                "cohort": "judgebench", "source": source, "domain": coarse_domain(source),
                "source_id": source_id, "pair_id": f'judge-{record["pair_id"]}',
                "question": question, "generator": generator,
                "input_limit": EXTENSION_MAX_TOKENS,
            }
            all_groups.append(row)
            if normalized(question) in excluded_questions:
                counts["calm_overlap"] += 1
                continue
            if (label not in ("A>B", "B>A") or not all(part.strip() for part in (question, a, b))
                    or normalized(a) == normalized(b) or "\ufffd" in text or STYLE_REQUEST.search(question)
                    or CONTROL_TOKEN.search(text) or VISUAL.search(text)):
                counts["invalid_text"] += 1
                continue
            correct = "fixed" if label == "A>B" else "target"
            if (arms := mirrored_arms(a, b, correct)) is None:
                counts["invalid_edit"] += 1
                continue
            row.update(reference=correct, answer_variants=arms)
            if not annotate_length(row, tokenizers):
                counts["over_token_limit"] += 1
                continue
            counts["eligible"] += 1
            rows.append(row)
    group_judgebench(all_groups)
    domains = defaultdict(set)
    for row in rows:
        domains[row["question_group"]].add(row["domain"])
    if any(len(group) > 1 for group in domains.values()):
        raise ValueError("A shared JudgeBench question spans multiple analysis domains")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate JudgeBench pair ID")
    return rows, dict(counts)


def calm_row(row):
    return {
        "pair_id": row["pair_id"], "source": row["source"], "source_id": row["source_id"],
        "question_group": row["source_id"], "domain": row["source"], "cohort": "calm",
        "question": row["question"], "input_limit": MAX_TOKENS,
        "answer_variants": {variant: [row["fixed_answer"], row["variants"][variant]]
                            for variant in STYLES},
    }


def data_manifest(root, stage, rows):
    rows = sorted(rows, key=lambda row: row["pair_id"])
    manifest = {
        "suite": "extension", "split": stage, "seed": SEED,
        "max_input_tokens": EXTENSION_MAX_TOKENS, "model_revisions": MODELS, "prompts": PROMPTS,
        "upstream_revisions": {"JudgeBench": JUDGEBENCH, "CALM": CALM, "GSM8K": GSM,
                               "MATH": MATH, "ScienceQA": SCIENCE},
        "rows": rows,
    }
    manifest["manifest_sha256"] = digest(manifest)
    path = manifest_path(stage, root)
    if path.exists() and read_json(path) != manifest:
        raise RuntimeError(f"The frozen {stage} manifest differs from the requested data")
    if not path.exists():
        write_json(path, manifest)
    return manifest["manifest_sha256"]


def prepare_extension(root):
    # The official Qwen3 tokenizers are byte-identical on these checkpoints.
    unique = {}
    for model in MODELS:
        directory = ASSETS / "models" / model
        names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        hashes = tuple(sha256((directory / name).read_bytes()) for name in names
                       if (directory / name).is_file())
        unique.setdefault(hashes, model)
    tokenizers = [load_tokenizer(model) for model in unique.values()]
    if any(not manifest_path(split).is_file() for split in ("development", "heldout")):
        from style_judges.data import prepare

        prepare(RESULTS)
    _, old_dev = get_manifest("development", RESULTS)
    _, old_test = get_manifest("heldout", RESULTS)
    old_ids = {row["source_id"] for row in (*old_dev, *old_test)}
    questions = {normalized(row["question"]) for row in (*old_dev, *old_test)}
    calm = []
    for item in candidate_rows(defaultdict(Counter)):
        if item["source_id"] in old_ids or normalized(item["question"]) in questions:
            continue
        row = calm_row(item)
        if annotate_length(row, tokenizers):
            calm.append(row)
            questions.add(normalized(row["question"]))

    judge, screening = judgebench_rows(tokenizers, questions)
    dev_ids = set(ranked_groups(judge, "pilot", DEV_GROUPS))
    if len(dev_ids) != DEV_GROUPS:
        raise RuntimeError("Insufficient eligible JudgeBench development groups")
    dev = [row for row in judge if row["question_group"] in dev_ids]
    heldout = [row for row in judge if row["question_group"] not in dev_ids]
    calm_dev = [calm_row(row) for row in old_dev]
    if not all(annotate_length(row, tokenizers) for row in calm_dev):
        raise RuntimeError("A previously verified CALM development prompt exceeds its token cap")
    data_manifest(root, "pilot", dev + calm_dev)
    write_json(root / "data" / "candidates.json", heldout + calm)
    selection = {
        "judgebench_screening": screening,
        "judgebench_hashes": {gen: sha256(judgebench_path(gen).read_bytes()) for gen in GENERATORS},
        "development_groups": len(dev_ids), "development_pairs": len(dev),
        "eligible_judgebench_pairs": len(judge), "eligible_heldout_judgebench_pairs": len(heldout),
        "eligible_heldout_judgebench_questions": len({row["question_group"] for row in heldout}),
        "eligible_fresh_calm_pairs": len(calm),
    }
    write_json(root / "data" / "screening.json", selection)
    print(json.dumps(selection, indent=2), flush=True)
