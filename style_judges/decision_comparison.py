"""Verify and combine the complete decision-model experiments."""
from style_judges import jev, decisions
from style_judges.analysis import write_csv
from style_judges.protocol import read_json, write_json, canonical_json, sha256


def main():
    baseline = read_json(jev.ROOT / 'manifest.json')
    metrics = {}
    summaries = {}
    for provider in ('jev', 'openai', 'gemini', 'perplexity', 'nimble'):
        root = jev.RESULTS / provider
        if read_json(root / 'manifest.json') != baseline:
            raise ValueError(f'{provider}: comparison manifest differs')
        backend = None if provider == 'jev' else decisions.Backend(provider)
        summaries[provider] = jev.analyze(root, backend)
        for row in jev.load_csv(root / 'analysis/metrics.csv'):
            key = tuple(row[k] for k in ('model','prompt','variant','metric'))
            if key in metrics and metrics[key] != row:
                raise ValueError('Inconsistent Qwen comparison rows')
            metrics[key] = row
    root = jev.RESULTS / 'decision_comparison'
    write_csv(root / 'metrics.csv', list(metrics.values()))
    write_json(root / 'summary.json', dict(manifest_sha256=sha256(canonical_json(baseline).encode()),
        pairs=len(baseline['rows']), providers=summaries,
        interpretation='CALM stability, not correctness. Model-service labels do not identify identical architectures or training.'))
    print('Combined results:', root)

if __name__ == '__main__':
    main()
