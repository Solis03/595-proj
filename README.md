# Style Sensitivity in Small LLM Judges

This CSE 595 project evaluates how redundancy, confidence language, and Markdown emphasis affect pairwise judgments by Qwen3-0.6B, 1.7B, and 4B. CALM measures verdict stability across 323 questions. JudgeBench measures accuracy and error transitions on 462 labeled pairs. The study compares both answer orders, two prompts, and a matched NF4 and BF16 subset.

## Reproduce on Windows

From PowerShell in the project directory, use a CUDA-capable NVIDIA driver and uv. The lockfile pins Python dependencies and the source downloader pins the three model snapshots, all dataset versions, and the report template.

~~~powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.tools\python'
$env:UV_PYTHON_BIN_DIR = Join-Path $PWD '.tools\bin'
uv sync --locked --managed-python --python 3.12
$python = '.\.venv\Scripts\python.exe'
& $python -m style_judges.sources data
& $python -m style_judges.sources models
& $python -m style_judges.extension fetch
& $python -m unittest discover -s tests -v
~~~

Run the evaluation and regenerate the unified report tables.

~~~powershell
& $python -m style_judges prepare
& $python -m style_judges run development nf4
& $python -m style_judges run heldout nf4
& $python -m style_judges analyze --precision nf4
& $python -m style_judges.extension prepare
& $python -m style_judges.extension run
& $python -m style_judges.extension analyze
& $python -m style_judges.extension_report
~~~

For a slow or interrupted full PyTorch transfer, run the following before installing the local wheel. The downloader retains completed ranges and validates the reconstructed file against the PyTorch index's SHA-256.

~~~powershell
uv sync --locked --managed-python --python 3.12 --no-install-package torch
$python = '.\.venv\Scripts\python.exe'
& $python -m style_judges.sources torch-wheel
uv pip install --python $python --no-deps '.\assets\wheels\torch-2.10.0+cu128-cp312-cp312-win_amd64.whl'
~~~

The supervisor resumes from saved judgment keys and records GPU worker time. The report generator reads both CALM judgment exports and the JudgeBench results.

## Results and report

- [Final ACL report](output/pdf/final_project_extended.pdf) covers verdict stability, correctness, and the effects of prompts, decoding, and precision.
- [Editable report source](report/final_project_extended.tex) uses the ACL final style. Author information, team ID, and contribution details are marked for the team to replace.
- [Judgment exports and analysis](results) include model scores, metric estimates, and runtime summaries. The [joint CALM and JudgeBench outputs](results/extension) also record source-question groups and JudgeBench answer labels. Preparation recreates the split manifests from pinned upstream data.

To build the report from the generated LaTeX fragments, run Tectonic from the report directory. The final report includes the result plot and works with compatible local LaTeX installations as well.

~~~powershell
Push-Location report
try { & ..\work\latex\tectonic.exe -X compile final_project_extended.tex }
finally { Pop-Location }
~~~

The source command `python -m style_judges.sources report` downloads the pinned ACL files and the Windows Tectonic zip into `work/latex`. Extract the executable into the same directory before compiling.

## Sources

CALM supplies candidate answers to questions from GSM8K, MATH, and ScienceQA. JudgeBench supplies answer pairs and published correctness labels. The judges receive only the question and two candidate answers. The report bibliography lists the dataset and benchmark papers.

## Optional Jev comparison (CPU/API only)

`style_judges.jev` implements the proposal's optional 40-pair CALM comparison.
It needs Python 3.12+ and NumPy for analysis/tests; it does not import PyTorch,
load Qwen weights, or require CUDA. A separate environment is useful on macOS:

```sh
python3.12 -m venv .venv-jev
.venv-jev/bin/python -m pip install 'numpy>=2,<3'
.venv-jev/bin/python -m style_judges.jev prepare
# Set TYPESAFE_API_KEY_SIXIAN in your shell using your credential manager.
.venv-jev/bin/python -m style_judges.jev run --max-calls 20
# Resume to complete the matrix (already saved calls are skipped).
.venv-jev/bin/python -m style_judges.jev run --max-calls 800
.venv-jev/bin/python -m style_judges.jev analyze
```

The default pinned model is `jev-1.14.0`; use `--model` for another explicit
version and `--output` for a separate experiment. `--key-env` selects a credential
variable; the default uses only `TYPESAFE_API_KEY_SIXIAN`, with no fallback to
another account. No credentials are saved. A bare `run` makes at most one new
request. Failures stop the run; there are no automatic paid retries. If a response
is lost after the server processes it, resuming may charge for that request again.

Preparation downloads the pinned CALM factual file and reconstructs the original
held-out inputs using the archived selection metadata (all targets were answer2;
all verbosity edits were verbatim duplication). It selects 12 GSM8K, 12 MATH, and
16 ScienceQA pairs by a fixed hash ranking, without looking at verdicts. The
frozen manifest embeds all three Qwen models' matching judgments and provenance.
This is a newly fixed subset, not an assertion that the proposal named these IDs.
It relies on the archived selection metadata; it does not revalidate Qwen token
hashes or rerun the original eligibility screening.

The matrix contains 40 pairs × 5 conditions (original, three edits, unchanged
repeat) × 2 answer orders × 2 prompts = **800 requests**. Each request contains
one Choice with A/B/T criteria and the existing prompt verbatim. Its state retains
the existing question/answer text layout, without Qwen's chat template. This is a
matched semantic protocol, not identical tokenization or decoding across models.
Repeats make fresh API requests even though their inputs equal the originals.

Results are checkpointed to `results/jev/judgments.sqlite`, with the request,
response probabilities, served version, token usage, and end-to-end API latency.
A database write lock prevents concurrent workers from duplicating saved jobs.
Changing the manifest, model, account-variable name, or prompts requires a new
output directory. Invalid replies remain saved for inspection and stop analysis.

Analysis requires the complete matrix. It exports `analysis/judgments.csv`,
`analysis/metrics.csv`, and `analysis/summary.json`: same-order flip rates,
repeat disagreement, tie rates, and order consistency for Jev and all three Qwen
models on exactly the same 40 pairs. Intervals resample paired questions within
source strata (10,000 draws). Jev probabilities are stored directly, never as
Qwen logits. No correctness score is inferred from unlabeled CALM data, and no
JudgeBench, pooled-rule, or forced-choice Jev result is claimed. API latency
includes network overhead and is not directly equivalent to local GPU timing.
The cost estimate uses $0.042/million input tokens, output tokens free (documentation
checked 2026-11-28); it is not an invoice and excludes lost responses.

API contracts and pricing: [Choice](https://docs.typesafe.ai/primitives/choice),
[System One](https://docs.typesafe.ai/api/system-one),
[Models](https://docs.typesafe.ai/models).

## OpenAI and Gemini Decisions comparison

The same frozen `results/jev/manifest.json` is reused without resampling:

```sh
python -m style_judges.decisions openai prepare
python -m style_judges.decisions gemini prepare
python -m style_judges.decisions openai run --max-calls 800
python -m style_judges.decisions gemini run --max-calls 800
python -m style_judges.decisions openai analyze
python -m style_judges.decisions gemini analyze
```

Use the CPU environment above (NumPy is only needed for analysis/tests).
Each provider has 800 real requests, including fresh unchanged-input repeats.
Results go to `results/openai` and `results/gemini`. The adapters reuse the Jev
ledger, frozen input order, completeness checks, and bootstrap analysis. They
send the exact same prompt, question/answer text, and option descriptions as Jev;
only the provider's required request envelope changes. OpenAI Choice probabilities
are converted from a list to a keyed mapping; Gemini's probability map is retained.
Native responses and all request bodies are exported to `analysis/raw_responses.json`.
Refusals or malformed replies stop the run and remain saved for inspection;
they are never silently converted to ties or discarded from denominators.

OpenAI uses `gpt-6-luna` at `/v1/decisions`; Gemini uses `gemini-decision-1` at
`/v1beta/decisions`. These are named service models, not immutable local checkpoint
hashes. OpenAI's served model field is checked; Gemini's response does not expose
a served version, so its model identity is explicitly marked as requested-only.
OpenAI prompt caching is disabled via explicit mode without breakpoints; Gemini
Decisions does not offer context caching. There are no automatic API retries.
Latency includes network overhead and is measured with one sequential request per
provider (providers may run concurrently).

Credentials are read from `OPENAI_API_KEY_SIXIAN_EXP20261228` and
`GEMINI_API_KEY_SIXIAN_EXPUNK`. On macOS, if unset, the adapters read their matching
sixian keychain service directly into process memory. Keys are never stored in
results, command arguments, or source. Other platforms should set these environment
variables. The code does not fall back to a different user's account.

Both adapters estimate cost using the published $0.10/million input tokens
(checked 2026-11-28); outputs are uncharged. Gemini may instead use free-tier quota,
so these are list-price equivalents, not invoices. Setup probes are outside the
800-request matrices and their token usage is not included in experiment summaries.

Sources: [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions),
[Gemini Decisions](https://aistudio.google.com/docs/decisions-api),
[Gemini pricing](https://aistudio.google.com/docs/pricing#gemini-decision).

After all three runs are complete, `python -m style_judges.decision_comparison`
revalidates their manifests and ledgers and writes the combined table and summaries
to `results/decision_comparison/`. It rejects mismatched samples or inconsistent
Qwen rows rather than merging them. The November 28 OpenAI run required explicit
recovery from a transient HTTP 503; recovery events are recorded in
`results/openai/recovery_log.jsonl`. Successful saved requests were not repeated;
server-side work on failed requests is unknown and excluded from cost estimates.

## Perplexity Decider v1.3 and Bespoke Nimble

These providers use the same frozen 40-pair manifest and the same 800-request
matrix. Prepare, run, and analyze them with:

```sh
python -m style_judges.decisions perplexity prepare
python -m style_judges.decisions nimble prepare
python -m style_judges.decisions perplexity run --max-calls 800
python -m style_judges.decisions nimble run --max-calls 800
python -m style_judges.decisions perplexity analyze
python -m style_judges.decisions nimble analyze
python -m style_judges.decision_comparison
```

Perplexity is fixed to `pplx-decider-v1.3`, with no alias, fallback to v1.2, or
web grounding. It uses `https://api.perplexity.ai/v1/decisions` and
`PERPLEXITY_API_KEY`. Its response model name only echoes the request; the ledger
marks this as `request_echo_only`, not independent checkpoint verification.
The documented isolation refers to a question not seeing *other questions* in
v1.3; the content and the current question's instructions remain visible. This
experiment sends one question per request throughout.

Nimble uses `nimble-v4.1-max` at
`https://api.bespokelabs.ai/v1/nimble/systemone`, with `BESPOKE_API_KEY`.
Its configured keychain fallback is the user's sixian account, as for Perplexity.
Nimble runs serially (one in flight), below the four-request free-credit limit.
It uses a $0.90 experiment ceiling: before each new request, saved token costs
plus the maximum charge for one 262,144-token request must fit that ceiling.
This guards this experiment's spending, not other concurrent account activity.
No credit purchases are made. HTTP 402 stops the run.

Both providers use the original Jev `state`/`questions` envelope and A/B/T option
descriptions unchanged. All raw responses are retained. Nimble can read question
IDs, so the neutral `verdict` ID is kept fixed, never derived from labels or
conditions. The runner uses $0.02/million input tokens for Perplexity and
$0.04/million for Nimble, outputs free (checked 2026-11-28). These estimates exclude
any unobserved provider-side work on failed requests.

References: [Perplexity Decisions](https://docs.perplexity.ai/docs/decisions/decisions),
[Nimble models and endpoint](https://docs.bespokelabs.ai/nimble/overview),
[Nimble pricing](https://docs.bespokelabs.ai/nimble/pricing),
[Nimble limits](https://docs.bespokelabs.ai/nimble/limits-and-errors).
