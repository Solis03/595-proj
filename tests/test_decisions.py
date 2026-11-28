import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from style_judges import decisions, jev
from style_judges.protocol import write_json, read_json
import test_jev


class DecisionsTests(unittest.TestCase):
    def native(self, provider):
        a = dict(type='choice', name='verdict', choice='A', probabilities={'A':.7,'B':.2,'T':.1}, confidence=.7)
        if provider in ('perplexity', 'nimble'):
            return dict(model=decisions.PROVIDERS[provider]['model'],answers={'verdict':a},usage={'input_tokens':100,'output_tokens':0})
        if provider == 'gemini':
            return dict(answers={'verdict':a},usage={'total_input_tokens':100,'total_output_tokens':0})
        a['probabilities'] = [dict(value=k,probability=v) for k,v in a['probabilities'].items()]
        return dict(model='gpt-6-luna',answers=[a],usage={'input_tokens':100,'output_tokens':0})

    def test_identical_content_and_criteria(self):
        row = test_jev.JevTests().manifest()['rows'][0]
        base = jev.request_body(row,'verbosity','target_first','rubric',jev.MODEL)
        for provider in decisions.PROVIDERS:
            b = decisions.Backend(provider)
            body = b.request_body(row,'verbosity','target_first','rubric',b.spec['model'])
            self.assertEqual(base['state'],body.get('input',body.get('state')))
            if provider != 'openai':
                self.assertEqual(base['questions'],body['questions'])
            else:
                q = body['questions'][0]
                self.assertEqual(q['instructions'],base['questions']['verdict']['instructions'])
                self.assertEqual({v['value']:v['description'] for v in q['choices']},base['questions']['verdict']['criteria'])

    def test_response_mapping_and_invalid_options(self):
        for provider in decisions.PROVIDERS:
            b = decisions.Backend(provider)
            native = self.native(provider)
            r = b.normalize(native)
            self.assertEqual(r['raw_response'],native)
            self.assertEqual(jev.validate_response(r,b.spec['model']),'A')
        native = self.native('openai')
        native['answers'][0]['probabilities'][1]['value']='A'
        with self.assertRaises(ValueError):
            decisions.Backend('openai').normalize(native)
        native = self.native('gemini')
        native['answers']['verdict']['type']='refusal'
        with self.assertRaises(ValueError):
            decisions.Backend('gemini').normalize(native)

    def test_both_provider_ledgers_resume_and_analysis(self):
        for provider in decisions.PROVIDERS:
            b = decisions.Backend(provider)
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                write_json(root/'manifest.json',test_jev.JevTests().manifest())
                with patch.dict('os.environ',{b.spec['key_env']:'fake'}), patch.object(b,'post',return_value=b.normalize(self.native(provider))) as post:
                    jev.run(root,b.spec['model'],60,b.spec['key_env'],b)
                    self.assertEqual(jev.run(root,b.spec['model'],60,b.spec['key_env'],b)['new_calls'],0)
                    self.assertEqual(post.call_count,60)
                summary=jev.analyze(root,b)
                self.assertEqual(summary['model'],b.spec['model'])
                self.assertAlmostEqual(summary['estimated_cost_usd'],6000*b.config()['input_usd_per_million']/1e6)
                self.assertEqual(len(read_json(root/'analysis/raw_responses.json')),60)

    def test_perplexity_requires_v13_and_marks_model_echo(self):
        b = decisions.Backend('perplexity')
        row = test_jev.JevTests().manifest()['rows'][0]
        with self.assertRaises(ValueError):
            b.request_body(row,'original','fixed_first','brief','pplx-decider-v1.2')
        body = b.request_body(row,'original','fixed_first','brief',b.spec['model'])
        self.assertNotIn('mode',body)
        self.assertEqual(b.normalize(self.native('perplexity'))['model_identity_source'],'request_echo_only')

    def test_nimble_budget_prevents_next_request(self):
        b = decisions.Backend('nimble')
        config = b.config()
        config['budget_usd'] = .000001
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root/'manifest.json',test_jev.JevTests().manifest())
            with patch.dict('os.environ',{b.spec['key_env']:'fake'}), patch.object(b,'config',return_value=config), patch.object(b,'post') as post:
                with self.assertRaisesRegex(RuntimeError,'budget reached'):
                    jev.run(root,b.spec['model'],60,b.spec['key_env'],b)
                post.assert_not_called()
