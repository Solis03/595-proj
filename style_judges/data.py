"""Construct fixed CALM pairs and source-backed preference labels."""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import json
import re
import unicodedata

from style_judges.protocol import (
    MAX_TOKENS, ORDERS, PROMPTS, RESULTS, SEED, STYLES, canonical_json, load_tokenizer,
    manifest_path, render, sha256, token_ids, write_json,
)
from style_judges.sources import ASSETS, CALM, GSM, MATH, MODELS, SCIENCE


QUOTAS = {
    "development": {"GSM8K": 6, "MATH": 6, "ScienceQA": 8},
    "heldout": {"GSM8K": 31, "MATH": 29, "ScienceQA": 40},
}
CONFIDENCE_PREFIX = "Certainly.\n\n"
CONTROL_TOKEN = re.compile(r"<\|[^>]+\|>|</?think>", re.I)
VISUAL = re.compile(r"\[/?asy\]|\\includegraphics|<(?:image|img)\b", re.I)
STYLE_REQUEST = re.compile(
    r"\b(?:in (?:one|a single) sentence|in (?:bullet points|Markdown|JSON)|"
    r"reply (?:only|in)|answer only with|give only the)\b", re.I,
)
PROSE = re.compile(r"(?m)^[ \t]*(?P<span>[^\W\d_]+(?:['’][^\W\d_]+)?"
                   r"(?: [^\W\d_]+(?:['’][^\W\d_]+)?){1,5})(?![\w'’])")
NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[-+]?\d+)?"
NUM_TOKEN = re.compile(rf"(?<![\w.,]){NUM}(?![\d,/]|[.]\d)")
NARRATIVE_FINAL = re.compile(
    r"^\W*(?:therefore|consequently|thus|hence|so|altogether|in total|in conclusion)\b", re.I
)
EXPLICIT_FINAL = re.compile(
    rf"^\W*(?:(?:therefore|thus|hence|so)[,\s]+)?(?:the\s+)?"
    rf"(?:final answer|answer|solution|result|remainder|conclusion)"
    rf"\s*(?:is\b|equals\b|:|=)\s*[*$\s]*(?P<value>{NUM})(?![\d,/]|[.]\d)",
    re.I,
)


def normalized(text):
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})).split())


def box_values(text):
    values = []
    for marker in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth = 1
        for index in range(marker.end(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[marker.end():index])
                    break
    return values


def scalar(text):
    clean = re.sub(r"[\s,$]", "", text.replace("−", "-"))
    clean = re.sub(r"\\(?:left|right|,|!|%)", "", clean)
    latex = re.fullmatch(r"([+-]?)\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", clean)
    if latex:
        sign, num, den = latex.groups()
        clean = ("-" if sign == "-" else "") + f"{num}/{den}"
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d+/[+-]?\d+)", clean):
        return None
    try:
        numerator, slash, denominator = clean.partition("/")
        return Fraction(numerator) / int(denominator) if slash else Fraction(numerator)
    except (ValueError, ZeroDivisionError):
        return None


def numeric_conclusion(answer):
    # Multiple distinct explicit conclusions leave the preference unverified.
    if boxes := box_values(answer):
        numbers = {scalar(value) for value in boxes}
        return next(iter(numbers)) if len(numbers) == 1 else None
    last = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())[-1]
    if explicit := EXPLICIT_FINAL.search(last):
        return scalar(explicit["value"])
    if NARRATIVE_FINAL.search(last) and "=" not in last:
        numbers = [scalar(match[0]) for match in NUM_TOKEN.finditer(last)]
        if len(numbers) == 1:
            return numbers[0]
    return None


def choice_conclusion(answer, choices):
    text = normalized(answer).casefold()
    text = re.sub(r"[*_\x60\\]", "", text)
    matches = set()
    for index, choice in enumerate(choices):
        choice = normalized(choice).casefold().strip(" .")
        escaped = re.escape(choice)
        if choice in {"yes", "no"} and re.match(rf"^{escaped}(?=\s*[,.!:]|$)", text):
            matches.add(index)
        if re.search(
            rf"(?:\b(?:correct (?:answer|choice|option)|(?:final )?answer|choice)"
            rf"\s*(?:is\b|:)|^\s*|\b(?:therefore|thus|so)[,:])\s*[\"']?{escaped}"
            rf"[\"']?(?=[.!?,;:]|\s+(?:is|because|as)\b|$)", text,
        ):
            matches.add(index)
    for match in re.finditer(
        r"\b(?:correct (?:answer|choice|option)|(?:final )?answer)\s*(?:is\b|:)"
        r"\s*(?:option\s+)?\(?([a-z])\)?(?=[.)!:]|\s*[-–]|\s*$)", text,
    ):
        index = ord(match[1]) - ord("a")
        if index < len(choices):
            matches.add(index)
    return next(iter(matches)) if len(matches) == 1 else None


def reference_for(source, official, a1, a2):
    if source == "ScienceQA":
        extracted = [choice_conclusion(answer, official["choices"]) for answer in (a1, a2)]
        gold = official["gold"]
    else:
        extracted = [numeric_conclusion(answer) for answer in (a1, a2)]
        gold = scalar(official["gold"] or "")
    if gold is None:
        state = "unsupported_source_answer"
    elif None in extracted:
        state = "ambiguous_candidate_conclusion"
    elif sum(value == gold for value in extracted) != 1:
        state = "both_match" if extracted[0] == gold else "neither_matches"
    else:
        state = "verified"
        if source != "ScienceQA":
            nonmatching = (a1, a2)[int(extracted[0] == gold)]
            tail = " ".join(re.split(r"(?<=[.!?])\s+|\n+", nonmatching.strip())[-2:])
            if any(scalar(match[0]) == gold for match in NUM_TOKEN.finditer(tail)):
                state = "ambiguous_candidate_conclusion"
    winner = f"answer{extracted.index(gold) + 1}" if state == "verified" else None
    evidence = [str(value) if isinstance(value, Fraction) else value for value in extracted]
    return winner, state, evidence


def source_index():
    import pyarrow.parquet as pq

    directory = ASSETS / "data"
    index = defaultdict(dict)
    for path in sorted(directory.glob("gsm_*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            record = json.loads(line)
            entry = {
                "id": f"GSM8K:{path.stem}:{line_no}",
                "gold": record["answer"].rsplit("####", 1)[-1].strip(),
                "source_solution": record["answer"],
            }
            index["GSM8K", normalized(record["question"])][entry["id"]] = entry
    for path in sorted((directory / "math").glob("*/*.parquet")):
        rows = pq.read_table(path, columns=["problem", "solution"]).to_pylist()
        for line_no, record in enumerate(rows, 1):
            boxes = box_values(record["solution"])
            entry = {
                "id": f"MATH:{path.parent.name}:{path.name.split('-')[0]}:{line_no}",
                "gold": boxes[-1] if boxes else None,
                "source_solution": record["solution"],
            }
            index["MATH", normalized(record["problem"])][entry["id"]] = entry
    for item, record in json.loads((directory / "scienceqa.json").read_text(encoding="utf-8")).items():
        if record["image"]:
            continue
        stem = record["question"]
        choices = record["choices"]
        entry = {
            "id": f"ScienceQA:{item}",
            "gold": record["answer"],
            "choices": choices,
            "source_solution": record["solution"],
        }
        for prompt in {stem, stem + "\n" + (record["hint"] or "")}:
            key = "ScienceQA", normalized(prompt + " Choices: " + ", ".join(choices))
            index[key][entry["id"]] = entry
    return index


def sentences(text):
    return re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip())


def released_repetition(original, released):
    if not released or len(released.split()) <= len(original.split()):
        return False
    at = released.find(original)
    if at < 0:
        return False
    allowed = set(sentences(original))
    extra = (released[:at], released[at + len(original):])
    return all(not part.strip() or all(sentence in allowed for sentence in sentences(part))
               for part in extra)


def apply_edits(target, released):
    span = PROSE.search(target)
    if not span or target.startswith(CONFIDENCE_PREFIX):
        return None
    start, stop = span.span("span")
    use_released = released_repetition(target, released)
    versions = {
        "original": target,
        "verbosity": released if use_released else target + "\n\n" + target,
        "confidence": CONFIDENCE_PREFIX + target,
        "formatting": target[:start] + "**" + target[start:stop] + "**" + target[stop:],
    }
    words = len(target.split())
    if any(10 * abs(len(versions[style].split()) - words) > words
           for style in ("confidence", "formatting")):
        return None
    evidence = {"verbosity": "CALM verbatim repetition" if use_released else "verbatim duplicate",
                "formatting_span": [start, stop]}
    return versions, evidence


def validate_edits(row):
    variants = row["variants"]
    target = variants["original"]
    start, stop = row["edit_evidence"]["formatting_span"]
    formatted = variants["formatting"]
    assert formatted[start:start + 2] == formatted[stop + 2:stop + 4] == "**"
    assert formatted[:start] + formatted[start + 2:stop + 2] + formatted[stop + 4:] == target
    assert variants["confidence"].startswith(CONFIDENCE_PREFIX)
    assert variants["confidence"].removeprefix(CONFIDENCE_PREFIX) == target
    if row["edit_evidence"]["verbosity"] == "verbatim duplicate":
        assert variants["verbosity"] == target + "\n\n" + target
    else:
        assert released_repetition(target, variants["verbosity"])
    assert len({*variants.values()}) == len(STYLES)
    for style in ("confidence", "formatting"):
        assert 10 * abs(len(variants[style].split()) - len(target.split())) <= len(target.split())


def pair_key(record):
    return tuple(record[key] for key in ("data_resource", "question", "answer1", "answer2"))


def candidate_rows(counts):
    index = source_index()
    directory = ASSETS / "data"
    raw = json.loads((directory / "calm_fact.json").read_text(encoding="utf-8"))
    release = json.loads((directory / "calm_verbosity.json").read_text(encoding="utf-8"))
    released_edits = {pair_key(record): record["answer2_longer"] for record in release}
    seen = set()
    for dataset_index, record in enumerate(raw):
        source, question, a1, a2 = pair_key(record)
        counts[source]["upstream"] += 1
        key = normalized(question)
        text = "\n".join((question, a1, a2))
        if (not all(value.strip() for value in (question, a1, a2))
                or "\ufffd" in text or VISUAL.search(text) or CONTROL_TOKEN.search(text)
                or STYLE_REQUEST.search(question) or normalized(a1) == normalized(a2)):
            counts[source]["ineligible_text"] += 1
            continue
        matches = index[source, key]
        if len(matches) != 1:
            counts[source]["no_unique_text_source_match"] += 1
            continue
        official = next(iter(matches.values()))
        if official["id"] in seen:
            counts[source]["duplicate_source_question"] += 1
            continue
        winner, state, extracted = reference_for(source, official, a1, a2)
        fixed, target, target_id = (a2, a1, "answer1") if winner == "answer2" else (a1, a2, "answer2")
        released = released_edits.get(pair_key(record)) if target_id == "answer2" else None
        if released and CONTROL_TOKEN.search(released):
            released = None
        edited = apply_edits(target, released)
        if edited is None:
            counts[source]["ineligible_edit"] += 1
            continue
        variants, evidence = edited
        seen.add(official["id"])
        row = {
            "pair_id": f"calm-{dataset_index:03d}",
            "source_id": official["id"],
            "source": source,
            "question": question,
            "fixed_answer": fixed,
            "target_original_id": target_id,
            "variants": variants,
            "reference": "fixed" if winner else None,
            "reference_status": state,
            "reference_original_id": winner,
            "extracted_conclusions_answer1_answer2": extracted,
            "source_gold": official["gold"],
            "source_solution": official["source_solution"],
            "edit_evidence": evidence,
        }
        if "choices" in official:
            row["source_choices"] = official["choices"]
        validate_edits(row)
        yield row


def check_prompt_lengths(row, tokenizers):
    longest = 0
    for tokenizer, style, order, prompt in product(tokenizers, STYLES, ORDERS, PROMPTS):
        formatted = render(tokenizer, row, style, order, prompt)
        longest = max(longest, len(token_ids(tokenizer, formatted)))
    return longest


def sample_rows(eligible, split, exclude):
    def rank(row):
        digest = sha256(f"{SEED}:{split}:{row['source_id']}".encode())
        if split == "development":
            return (False, digest)
        return (row["reference"] is None, digest)

    quotas = QUOTAS[split]
    remaining = [row for row in eligible if row["source_id"] not in exclude]
    remaining.sort(key=rank)
    chosen = []
    for source, quota in quotas.items():
        chosen.extend([row for row in remaining if row["source"] == source][:quota])
    chosen_ids = {row["source_id"] for row in chosen}
    shortage = sum(quotas.values()) - len(chosen)
    if shortage:
        chosen.extend([row for row in remaining if row["source_id"] not in chosen_ids][:shortage])
    if len(chosen) != sum(quotas.values()) or set(quotas) != {row["source"] for row in chosen}:
        raise RuntimeError(f"Insufficient eligible source coverage for {split}")
    return sorted(chosen, key=lambda row: row["pair_id"])


def prepare(root=RESULTS):
    tokenizers = [load_tokenizer(model) for model in MODELS]
    eligible = []
    counts = defaultdict(Counter)
    for row in candidate_rows(counts):
        longest = check_prompt_lengths(row, tokenizers)
        if longest > MAX_TOKENS:
            counts[row["source"]]["over_token_limit"] += 1
            continue
        row["max_prompt_tokens"] = longest
        counts[row["source"]]["eligible"] += 1
        if row["reference"]:
            counts[row["source"]]["eligible_with_reference"] += 1
        eligible.append(row)
    exclude = set()
    seen_questions = set()
    summaries = {}
    for split, quotas in QUOTAS.items():
        rows = sample_rows(eligible, split, exclude)
        exclude.update(row["source_id"] for row in rows)
        questions = {normalized(row["question"]) for row in rows}
        assert len(questions) == len(rows) and seen_questions.isdisjoint(questions)
        seen_questions.update(questions)
        sources = Counter(row["source"] for row in rows)
        gold_sources = Counter(row["source"] for row in rows if row["reference"])
        manifest = {
            "split": split,
            "seed": SEED,
            "upstream_revisions": {"CALM": CALM, "GSM8K": GSM, "MATH": MATH, "ScienceQA": SCIENCE},
            "model_revisions": MODELS,
            "prompts": PROMPTS,
            "max_input_tokens": MAX_TOKENS,
            "rows": rows,
        }
        manifest["manifest_sha256"] = sha256(canonical_json(manifest).encode("utf-8"))
        write_json(manifest_path(split, root), manifest)
        summaries[split] = {
            "rows": len(rows), "sources": dict(sources), "verified_sources": dict(gold_sources),
            "reference_status": dict(Counter(row["reference_status"] for row in rows)),
            "verbosity_provenance": dict(Counter(row["edit_evidence"]["verbosity"] for row in rows)),
            "target_answer": dict(Counter(row["target_original_id"] for row in rows)),
            "requested_quotas": quotas,
            "manifest_sha256": manifest["manifest_sha256"],
        }
    summaries["screening"] = {source: dict(detail) for source, detail in counts.items()}
    write_json(root / "data" / "selection.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)
