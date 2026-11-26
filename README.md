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
