# Style Sensitivity in Small LLM Judges

This CSE 595 project tests whether redundancy, confidence language, and Markdown emphasis change pairwise judgments on CALM questions. The experiment uses official Qwen3 0.6B, 1.7B, and 4B checkpoints, two prompts, both answer orders, and an unchanged repeat. The held-out run comprises 6,000 judgments.

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
& $python -m unittest discover -s tests -v
& $python -m style_judges prepare
& $python -m style_judges run development nf4
& $python -m style_judges run heldout nf4
& $python -m style_judges analyze --precision nf4
& $python -m style_judges.report
~~~

For a slow or interrupted full PyTorch transfer, run the following before installing the local wheel. The downloader retains completed ranges and validates the reconstructed file against the PyTorch index's SHA-256.

~~~powershell
uv sync --locked --managed-python --python 3.12 --no-install-package torch
$python = '.\.venv\Scripts\python.exe'
& $python -m style_judges.sources torch-wheel
uv pip install --python $python --no-deps '.\assets\wheels\torch-2.10.0+cu128-cp312-cp312-win_amd64.whl'
~~~

Use the same precision for both phases and analysis. If NF4 fails the CUDA preflight, the supported fallback is BF16 for all three checkpoints. The supervisor resumes from persisted decision keys and records process elapsed time in a shared 24 GPU-hour ledger. Held-out inference checks the corresponding complete development run and its throughput forecast before launch.

## Results and report

- [Data selection](results/data/selection.json) records source counts and manifest hashes. The preparation command recreates the exact split manifests from the pinned upstream snapshots.
- [Experiment results](results) contains the CSV judgment exports and timing and memory logs. Local inference also writes a resumable SQLite ledger and a GPU time ledger.
- [Analysis](results/analysis) contains clustered estimates, transitions, and runtime summaries. The analysis command also recreates the case-study data locally.
- [Final ACL report](output/pdf/final_project.pdf) contains the measured comparison, case studies, and limitations.
- [Report source](report/final_project.tex) uses the ACL final style. Author information, team ID, contribution statement, and code URL are marked for the team to replace.

To build the report from the generated LaTeX fragments, run Tectonic from the report directory. The final report includes the result plot and works with compatible local LaTeX installations as well.

~~~powershell
Push-Location report
try { & ..\work\latex\tectonic.exe -X compile final_project.tex }
finally { Pop-Location }
~~~

The source command `python -m style_judges.sources report` downloads the pinned ACL files and the Windows Tectonic zip into `work/latex`. Extract the executable into the same directory before compiling.

## Sources

The CALM release supplies the original answer pairs. Original GSM8K, MATH, and ScienceQA solutions are used only by the data preparation step. Model inputs contain the question and two candidate answers. The report bibliography lists the dataset and benchmark papers.
