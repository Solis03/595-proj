"""Decision-model provider adapters for the frozen CALM/Jev experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from style_judges import jev
from style_judges.protocol import canonical_json, read_json, write_json

PROVIDERS = {
    'perplexity': dict(model='pplx-decider-v1.3', endpoint='https://api.perplexity.ai/v1/decisions',
                       key_env='PERPLEXITY_API_KEY', service='llm.perplexity.api_key.sixianli.c20261128.eunk'),
    'nimble': dict(model='nimble-v4.1-max', endpoint='https://api.bespokelabs.ai/v1/nimble/systemone',
                   key_env='BESPOKE_API_KEY', service='llm.bespoke.nimble.api_key.sixianli.c20261128.eunk'),
    'openai': dict(model='gpt-6-luna', endpoint='https://api.openai.com/v1/decisions',
                   key_env='OPENAI_API_KEY_SIXIAN_EXP20261228',
                   service='llm.openai.api_key.sixianli.c20261128.e20261228'),
    'gemini': dict(model='gemini-decision-1', endpoint='https://generativelanguage.googleapis.com/v1beta/decisions',
                   key_env='GEMINI_API_KEY_SIXIAN_EXPUNK',
                   service='llm.google.gemini.api_key.sixianli.c20261128.eunk'),
}


class Backend:
    def __init__(self, provider):
        self.provider = provider
        self.spec = PROVIDERS[provider]

    def config(self):
        if self.provider in ('perplexity', 'nimble'):
            config = dict(provider=self.provider, endpoint=self.spec['endpoint'],
                          input_usd_per_million=.02 if self.provider == 'perplexity' else .04,
                          adapter_version=1,
                          model_identity_note=('Response model echoes requested name; not independent version verification.'
                                               if self.provider == 'perplexity' else
                                               'Requested nimble-v4.1-max; response model checked, no immutable checkpoint hash.'))
            if self.provider == 'nimble':
                config.update(budget_usd=.90, max_request_cost_usd=262144*.04/1e6)
            return config
        return dict(provider=self.provider, endpoint=self.spec['endpoint'], input_usd_per_million=.10,
                    adapter_version=1,
                    model_identity_note=('Requested named model; API reports served model, not an immutable checkpoint.'
                                         if self.provider == 'openai' else
                                         'Requested gemini-decision-1; API does not report a served checkpoint/version.'))

    def request_body(self, row, variant, order, prompt, model):
        if model != self.spec['model']:
            raise ValueError('Unexpected provider/model combination')
        base = jev.request_body(row, variant, order, prompt, model)
        if self.provider in ('perplexity', 'nimble'):
            return base
        question = base['questions']['verdict']
        body = dict(model=model, input=base['state'])
        if self.provider == 'gemini':
            body['questions'] = base['questions']
        else:
            body['questions'] = [dict(type='choice', name='verdict', instructions=question['instructions'],
                                      choices=[dict(value=k, description=v) for k, v in question['criteria'].items()])]
            # Avoid charging for cache writes and keep single-call timing interpretable.
            body['prompt_cache_options'] = {'mode': 'explicit'}
        return body

    def normalize(self, raw):
        model = self.spec['model']
        if self.provider in ('perplexity', 'nimble'):
            if raw.get('model') != model:
                raise ValueError('Unexpected response model name')
            answer = raw['answers']['verdict']
            usage = {k: raw['usage'][k] for k in ('input_tokens','output_tokens')}
            identity = 'request_echo_only' if self.provider == 'perplexity' else 'response_model'
        elif self.provider == 'gemini':
            answer = raw['answers']['verdict']
            if answer.get('type') != 'choice':
                raise ValueError('Gemini returned no valid Choice (possibly a refusal)')
            usage = dict(input_tokens=raw['usage']['total_input_tokens'], output_tokens=raw['usage']['total_output_tokens'])
            identity = 'requested_only'
        else:
            if raw.get('model') != model:
                raise ValueError('OpenAI served a different model')
            if len(raw['answers']) != 1:
                raise ValueError('Expected exactly one answer')
            a = raw['answers'][0]
            if a.get('type') != 'choice' or a.get('name') != 'verdict':
                raise ValueError('OpenAI returned no valid Choice (possibly a refusal)')
            if len(a['probabilities']) != 3 or len({p['value'] for p in a['probabilities']}) != 3:
                raise ValueError('Duplicate or missing probability options')
            answer = dict(a, probabilities={p['value']: p['probability'] for p in a['probabilities']})
            usage = {k: raw['usage'][k] for k in ('input_tokens','output_tokens')}
            identity = 'response_model'
        result = dict(model=model, answers={'verdict': answer}, usage=usage,
                      model_identity_source=identity, raw_response=raw)
        jev.validate_response(result, model)
        return result

    def post(self, body, key):
        headers = {'Content-Type': 'application/json'}
        headers.update({'x-goog-api-key': key} if self.provider == 'gemini' else {'Authorization': 'Bearer ' + key})
        req = Request(self.spec['endpoint'], data=canonical_json(body).encode(), headers=headers)
        try:
            with urlopen(req, timeout=60) as response:
                raw = json.load(response)
        except HTTPError as exc:
            # No retries or raw error bodies (which may contain account data).
            raise RuntimeError(f'{self.provider} HTTP {exc.code}; resume saved run after resolving the error') from None
        try:
            return self.normalize(raw)
        except (ValueError, KeyError, TypeError):
            # Persist native reply in the ledger; shared validator stops the run.
            return dict(model=self.spec['model'], answers={'verdict': {'type':'invalid'}},
                        usage={}, raw_response=raw)

    def load_key(self):
        env = self.spec['key_env']
        if not os.environ.get(env):
            result = subprocess.run(['security', 'find-generic-password', '-s', self.spec['service'], '-w'], capture_output=True, text=True)
            if result.returncode or not result.stdout.strip():
                raise RuntimeError(f'Set {env} or configure its keychain service')
            os.environ[env] = result.stdout.strip()
        return env


def prepare(provider, root):
    manifest = read_json(jev.ROOT / 'manifest.json')
    path = Path(root) / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Frozen manifest differs')
    write_json(path, manifest)
    return {'provider':provider, 'pairs':len(manifest['rows']), 'expected':len(list(jev.jobs(manifest, PROVIDERS[provider]['model'], Backend(provider))))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('provider', choices=PROVIDERS)
    p.add_argument('command', choices=['prepare', 'run', 'analyze'])
    p.add_argument('--max-calls', type=int, default=1)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    backend = Backend(args.provider)
    root = args.output or jev.RESULTS / args.provider
    if args.command == 'prepare':
        result = prepare(args.provider, root)
    elif args.command == 'analyze':
        result = jev.analyze(root, backend)
    else:
        result = jev.run(root, backend.spec['model'], args.max_calls, backend.load_key(), backend)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
