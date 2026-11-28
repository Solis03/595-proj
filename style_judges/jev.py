"""Optional CALM/Jev comparison. Uses stdlib HTTP; no CUDA or model downloads."""
import argparse
from contextlib import closing
import csv
from datetime import datetime, timezone
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from style_judges.data import apply_edits, pair_key, validate_edits
from style_judges.protocol import ORDERS, PROMPTS, VARIANTS, STYLES, RESULTS, canonical_json, sha256, write_json, read_json, verdict_identity
from style_judges.sources import CALM, MODELS

ROOT = RESULTS / 'jev'
MODEL = 'jev-1.14.0'
API = 'https://api.typesafe.ai/v1/systemone'
KEY_ENV = 'TYPESAFE_API_KEY_SIXIAN'


def load_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def prepare(root=ROOT, upstream=None):
    """Recover the original 100 held-out inputs, then select 40 without scores."""
    records = load_csv(RESULTS / 'heldout_judgments.csv')
    selection = read_json(RESULTS / 'data/selection.json')['heldout']
    # These assertions are necessary for reconstructing orientation/edits without gold data.
    if selection['target_answer'] != {'answer2': 100} or selection['verbosity_provenance'] != {'verbatim duplicate': 100}:
        raise ValueError('Original manifest required: saved experiment used other orientations/edits')
    if {r['manifest_hash'] for r in records} != {selection['manifest_sha256']}:
        raise ValueError('Qwen export/selection mismatch')
    if upstream is None:
        url = f'https://raw.githubusercontent.com/Y0oMu/LLM-Judge-Bias-Dataset/{CALM}/base_datasets/fact_related_dataset.json'
        with urlopen(url, timeout=60) as response:
            raw_bytes = response.read()
    else:
        raw_bytes = Path(upstream).read_bytes()
    raw = json.loads(raw_bytes)
    sources = {r['pair_id']: r['source'] for r in records}
    selected = []
    for source, quota in [('GSM8K', 12), ('MATH', 12), ('ScienceQA', 16)]:
        candidates = sorted((p for p, s in sources.items() if s == source),
                            key=lambda p: sha256(('jev-595-40-v1:' + p).encode()))
        if len(candidates) < quota:
            raise ValueError('Insufficient source coverage')
        for pid in candidates[:quota]:
            s, question, fixed, target = pair_key(raw[int(pid.split('-')[1])])
            if s != source:
                raise ValueError('Pinned source does not match export')
            edits = apply_edits(target, None)
            if edits is None:
                raise ValueError('Cannot reproduce edits')
            variants, evidence = edits
            row = dict(pair_id=pid, source=s, question=question, fixed_answer=fixed,
                       variants=variants, edit_evidence=evidence)
            validate_edits(row)
            selected.append(row)
    selected.sort(key=lambda r: r['pair_id'])
    ids = {r['pair_id'] for r in selected}
    reference = [r for r in records if r['pair_id'] in ids]
    expected = set(product(MODELS, ids, VARIANTS, ORDERS, PROMPTS))
    keys = [(r['model'], r['pair_id'], r['variant'], r['answer_order'], r['prompt']) for r in reference]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate Qwen comparison matrix')
    manifest = dict(protocol='calm-jev-40-v1', upstream_revision=CALM,
                    upstream_sha256=sha256(raw_bytes), qwen_manifest=selection['manifest_sha256'],
                    rows=selected, qwen=reference)
    destination = Path(root) / 'manifest.json'
    if destination.exists() and read_json(destination) != manifest:
        raise ValueError('Refusing to replace a different frozen manifest')
    write_json(destination, manifest)
    return {'pairs': len(selected), 'requests': len(selected) * len(VARIANTS) * len(ORDERS) * len(PROMPTS)}


def request_body(row, variant, order, prompt, model):
    target = row['variants']['original' if variant == 'repeat' else variant]
    a, b = (row['fixed_answer'], target) if order == 'fixed_first' else (target, row['fixed_answer'])
    return {'model': model,
            'state': f"Question:\n{row['question']}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\nVerdict:",
            'questions': {'verdict': {'type': 'choice', 'instructions': PROMPTS[prompt],
                                    'criteria': {'A': 'Answer A is better.', 'B': 'Answer B is better.',
                                                 'T': 'Neither answer is better on the stated criteria.'}}}}


def validate_response(response, model):
    if response.get('model') != model:
        raise ValueError('Served model differs from pinned version')
    answer = response['answers']['verdict']
    probs = answer['probabilities']
    if answer.get('type') != 'choice' or set(probs) != {'A', 'B', 'T'}:
        raise ValueError('Invalid Choice response')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
        raise ValueError('Invalid probabilities')
    if not math.isclose(sum(probs.values()), 1, abs_tol=1e-5):
        raise ValueError('Probabilities do not sum to one')
    label = answer['choice']
    if label not in probs or probs[label] < max(probs.values()) - 1e-8:
        raise ValueError('Choice disagrees with probabilities')
    for k in ('input_tokens', 'output_tokens'):
        value = response['usage'][k]
        if type(value) is not int or value < 0:
            raise ValueError('Invalid usage')
    return label


def post(body, key):
    # No implicit retries: a lost response may already have incurred a charge.
    req = Request(API, data=canonical_json(body).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f'TypeSafe HTTP {exc.code}; saved results retained. Retry the command later.') from None


def jobs(manifest, model, backend=None):
    for row, variant, order, prompt in product(manifest['rows'], VARIANTS, ORDERS, PROMPTS):
        body = (backend.request_body if backend else request_body)(row, variant, order, prompt, model)
        key = sha256(canonical_json([row['pair_id'], variant, order, prompt, body]).encode())
        yield key, row, variant, order, prompt, body


def run(root=ROOT, model=MODEL, max_calls=1, key_env=KEY_ENV, backend=None):
    if backend is None and not re.fullmatch(r'jev-\d+\.\d+\.\d+', model):
        raise ValueError('Use a pinned Jev version, not a moving alias')
    if max_calls < 1:
        raise ValueError('max-calls must be positive')
    key = os.environ.get(key_env)
    if not key:
        raise ValueError(f'Set {key_env}; credentials are never saved')
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    config = {'manifest_sha256': sha256(canonical_json(manifest).encode()), 'model': model,
              'key_env': key_env, 'endpoint': API, 'prompts': PROMPTS,
              'input_usd_per_million': 0.042, 'pricing_checked': '2026-11-28'}
    if backend:
        config.update(backend.config())
    # Exclusive SQLite transaction also prevents two workers from charging for the same job.
    with closing(sqlite3.connect(root / 'judgments.sqlite', timeout=1)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        saved = db.execute('SELECT value FROM config WHERE id=1').fetchone()
        if saved and json.loads(saved[0]) != config:
            raise ValueError('Run configuration changed; choose a separate output directory')
        db.execute('INSERT OR IGNORE INTO config VALUES (1, ?)', (json.dumps(config),))
        db.commit()
        called = 0
        budget = config.get('budget_usd')
        spent = (sum(json.loads(v)['response']['usage']['input_tokens'] for v, in db.execute('SELECT record FROM records'))
                 * config['input_usd_per_million'] / 1e6) if budget is not None else 0
        for jid, row, variant, order, prompt, body in jobs(manifest, model, backend):
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT record FROM records WHERE id=?', (jid,)).fetchone()
            if existing:
                validate_response(json.loads(existing[0])['response'], model)
                db.commit()
                continue
            if budget is not None and spent + config['max_request_cost_usd'] > budget:
                raise RuntimeError('Experiment budget reached; saved results retained')
            started = time.perf_counter()
            response = (backend.post if backend else post)(body, key)
            latency = (time.perf_counter() - started) * 1000
            # Persist even invalid replies for audit; analysis validates before use.
            record = dict(pair_id=row['pair_id'], variant=variant, answer_order=order, prompt=prompt,
                          request=body, response=response, latency_ms=latency,
                          timestamp=datetime.now(timezone.utc).isoformat())
            db.execute('INSERT INTO records VALUES (?,?)', (jid, json.dumps(record)))
            db.commit()
            validate_response(response, model)
            spent += response['usage']['input_tokens'] * config['input_usd_per_million'] / 1e6
            called += 1
            if called >= max_calls:
                break
        total = db.execute('SELECT count(*) FROM records').fetchone()[0]
    return {'new_calls': called, 'saved': total, 'expected': len(list(jobs(manifest, model, backend)))}


def analyze(root=ROOT, backend=None):
    import numpy as np
    from style_judges.analysis import cluster_draws, write_csv
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    with closing(sqlite3.connect(f'file:{root / "judgments.sqlite"}?mode=ro', uri=True)) as db:
        config = json.loads(db.execute('SELECT value FROM config WHERE id=1').fetchone()[0])
        records = {k: json.loads(v) for k, v in db.execute('SELECT id, record FROM records')}
    if config['manifest_sha256'] != sha256(canonical_json(manifest).encode()) or config['prompts'] != PROMPTS:
        raise ValueError('Manifest or prompts changed')
    expected = list(jobs(manifest, config['model'], backend))
    if set(records) != {j[0] for j in expected}:
        raise ValueError(f'Incomplete experiment: {len(records)}/{len(expected)}; no formal metrics written')
    verdicts = {}
    export = []
    for jid, row, variant, order, prompt, body in expected:
        rec = records[jid]
        if rec['request'] != body:
            raise ValueError('Stored request mismatch')
        label = validate_response(rec['response'], config['model'])
        verdicts[config['model'], row['pair_id'], variant, order, prompt] = verdict_identity(label, order)
        export.append(dict(pair_id=row['pair_id'], variant=variant, answer_order=order, prompt=prompt,
                           label=label, **{'probability_' + k: v for k, v in rec['response']['answers']['verdict']['probabilities'].items()},
                           latency_ms=rec['latency_ms'], **rec['response']['usage']))
    for r in manifest['qwen']:
        verdicts[r['model'], r['pair_id'], r['variant'], r['answer_order'], r['prompt']] = r['verdict']
    rows = manifest['rows']
    draws = cluster_draws(np.array([r['source'] for r in rows]))
    metrics = []
    for model, prompt in product([*MODELS, config['model']], PROMPTS):
        arrays = {v: np.array([[verdicts[model, r['pair_id'], v, o, prompt] for o in ORDERS] for r in rows]) for v in VARIANTS}
        for variant in VARIANTS:
            a = arrays[variant]
            values = {'tie_rate': (a == 'tie').mean(axis=1),
                      'order_consistency': (a[:, 0] == a[:, 1]).astype(float)}
            if variant != 'original':
                values['repeat_disagreement' if variant == 'repeat' else 'flip_rate'] = (a != arrays['original']).mean(axis=1)
            for metric, per_pair in values.items():
                low, high = np.percentile(per_pair[draws].mean(axis=1), [2.5, 97.5])
                metrics.append(dict(model=model, prompt=prompt, variant=variant, metric=metric,
                                    estimate=float(per_pair.mean()), ci_low=float(low), ci_high=float(high), pairs=len(rows)))
    tokens = sum(r['input_tokens'] for r in export)
    summary = dict(pairs=len(rows), calls=len(export), model=config['model'], input_tokens=tokens,
                   output_tokens=sum(r['output_tokens'] for r in export),
                   estimated_cost_usd=tokens * config['input_usd_per_million'] / 1e6,
                   api_latency_mean_ms=sum(r['latency_ms'] for r in export) / len(export),
                   note='CALM stability only; no gold correctness labels. API latency includes network. Cost excludes lost responses.')
    if backend:
        summary['cost_note'] = 'List-price input estimate; excludes cache write/read adjustments and free-tier eligibility, not an invoice.'
        summary['model_identity_note'] = config['model_identity_note']
        write_json(root / 'analysis/raw_responses.json', list(records.values()))
    write_csv(root / 'analysis/metrics.csv', metrics)
    write_csv(root / 'analysis/judgments.csv', export)
    write_json(root / 'analysis/summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--max-calls', type=int, default=1, help='New paid calls this invocation; full matrix is 800')
    parser.add_argument('--key-env', default=KEY_ENV)
    args = parser.parse_args()
    result = prepare(args.output) if args.command == 'prepare' else analyze(args.output) if args.command == 'analyze' else run(args.output, args.model, args.max_calls, args.key_env)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
