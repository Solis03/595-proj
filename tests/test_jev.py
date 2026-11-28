import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from style_judges import jev
from style_judges.protocol import ORDERS, PROMPTS, VARIANTS, write_json
from style_judges.sources import MODELS


class JevTests(unittest.TestCase):
    def manifest(self):
        rows = []
        qwen = []
        for i in range(3):
            target = 'The answer is twenty because these are the values in the question.'
            variants, evidence = jev.apply_edits(target, None)
            row = dict(pair_id=f'test-{i}', source='math', question='Which answer?',
                       fixed_answer='The answer is ten.', variants=variants, edit_evidence=evidence)
            rows.append(row)
            for model in MODELS:
                for v in VARIANTS:
                    for o in ORDERS:
                        for p in PROMPTS:
                            qwen.append(dict(model=model, pair_id=row['pair_id'], variant=v,
                                             answer_order=o, prompt=p, verdict='fixed'))
        return dict(rows=rows, qwen=qwen)

    @staticmethod
    def response():
        return dict(model=jev.MODEL, answers={'verdict': dict(type='choice', choice='A',
                    probabilities={'A': .7, 'B': .2, 'T': .1})}, usage=dict(input_tokens=100, output_tokens=10))

    def test_request_preserves_order_prompts_and_repeat(self):
        row = self.manifest()['rows'][0]
        a = jev.request_body(row, 'original', ORDERS[0], 'brief', jev.MODEL)
        b = jev.request_body(row, 'original', ORDERS[1], 'brief', jev.MODEL)
        self.assertNotEqual(a['state'], b['state'])
        self.assertEqual(a, jev.request_body(row, 'repeat', ORDERS[0], 'brief', jev.MODEL))
        self.assertEqual(a['questions']['verdict']['instructions'], PROMPTS['brief'])
        self.assertEqual(set(a['questions']['verdict']['criteria']), {'A', 'B', 'T'})

    def test_rejects_bad_probabilities_model_and_usage(self):
        for mutate in [lambda r: r.update(model='jev-latest'),
                       lambda r: r['answers']['verdict']['probabilities'].update(A=float('nan')),
                       lambda r: r['answers']['verdict'].update(choice='B'),
                       lambda r: r['usage'].update(input_tokens=-1)]:
            r = self.response()
            mutate(r)
            with self.assertRaises(ValueError):
                jev.validate_response(r, jev.MODEL)

    def test_resume_repeat_and_complete_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'manifest.json', self.manifest())
            with patch.dict('os.environ', {jev.KEY_ENV: 'test-only'}), patch.object(jev, 'post', return_value=self.response()) as post:
                self.assertEqual(jev.run(root, max_calls=1)['saved'], 1)
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    jev.analyze(root)
                self.assertFalse((root / 'analysis').exists())
                self.assertEqual(jev.run(root, max_calls=100)['saved'], 60)
                self.assertEqual(jev.run(root, max_calls=100)['new_calls'], 0)
                self.assertEqual(post.call_count, 60)  # repeats are actual new requests
            summary = jev.analyze(root)
            self.assertEqual(summary['calls'], 60)
            self.assertEqual(summary['input_tokens'], 6000)
            self.assertAlmostEqual(summary['estimated_cost_usd'], .000252)
            metrics = jev.load_csv(root / 'analysis/metrics.csv')
            flips = [float(r['estimate']) for r in metrics if r['metric'] == 'flip_rate']
            self.assertTrue(all(v == 0 for v in flips))
            rows = self.manifest()
            rows['rows'][0]['question'] = 'Changed'
            write_json(root / 'manifest.json', rows)
            with patch.dict('os.environ', {jev.KEY_ENV: 'test-only'}):
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    jev.run(root)
            with self.assertRaisesRegex(ValueError, 'Manifest'):
                jev.analyze(root)

    def test_directional_flip_and_pair_denominator(self):
        responses = [self.response() for _ in range(60)]
        # First pair, verbosity, first order, brief: one of six ordered decisions flips.
        responses[4]['answers']['verdict'].update(
            choice='B', probabilities={'A': .2, 'B': .7, 'T': .1})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'manifest.json', self.manifest())
            with patch.dict('os.environ', {jev.KEY_ENV: 'test-only'}), patch.object(jev, 'post', side_effect=responses):
                jev.run(root, max_calls=60)
            jev.analyze(root)
            metrics = jev.load_csv(root / 'analysis/metrics.csv')
            value = next(r for r in metrics if r['model'] == jev.MODEL and
                         r['prompt'] == 'brief' and r['variant'] == 'verbosity' and
                         r['metric'] == 'flip_rate')
            self.assertAlmostEqual(float(value['estimate']), 1 / 6)
            self.assertEqual(value['pairs'], '3')

    def test_invalid_saved_reply_cannot_be_silently_skipped(self):
        bad = self.response()
        bad['model'] = 'unexpected'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / 'manifest.json', self.manifest())
            with patch.dict('os.environ', {jev.KEY_ENV: 'test-only'}), patch.object(jev, 'post', return_value=bad) as post:
                with self.assertRaisesRegex(ValueError, 'Served model'):
                    jev.run(root)
                with self.assertRaisesRegex(ValueError, 'Served model'):
                    jev.run(root)
                self.assertEqual(post.call_count, 1)


if __name__ == '__main__':
    unittest.main()
