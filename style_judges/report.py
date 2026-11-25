"""Generate the ACL paper's measured content from validated experiment outputs."""

import argparse
import csv
from itertools import product
from pathlib import Path
import shutil

from style_judges.analysis import EDITS
from style_judges.protocol import PROMPTS, RESULTS, STYLES, read_json
from style_judges.sources import ACL, MODELS, ROOT


SIZES = {model: model.removeprefix("Qwen3-") for model in MODELS}
COMPACT_STYLES = ("Orig.", "Verb.", "Conf.", "Fmt.")


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def latex(text):
    escapes = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "#": r"\#",
        "%": r"\%", "&": r"\&", "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
        "’": "'", "“": "''", "”": "''", "–": "--", "—": "---",
    }
    return "".join(escapes.get(ch, ch) for ch in str(text))


def pct(value):
    return f"{100 * float(value):.1f}"


def signed(value):
    return f"{100 * float(value):+.1f}"


def table(headers, rows, caption, label, wide=False):
    env = "table*" if wide else "table"
    width = "@{}l" + "r" * (len(headers) - 1) + "@{}"
    body = [" & ".join(row) + r" \\" for row in rows]
    return "\n".join([
        rf"\begin{{{env}}}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
        rf"\begin{{tabular}}{{{width}}}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}}}", rf"\end{{{env}}}",
    ])


def generate(results, report):
    summary = read_json(results / "analysis" / "summary.json")
    assert summary["heldout_judgments"] == 6000 and summary["development_judgments"] == 1200
    all_metrics = load_csv(results / "analysis" / "metrics.csv")
    timing = load_csv(results / "analysis" / "latency_and_memory.csv")
    examples = read_json(results / "analysis" / "examples.json")
    selection = summary["selection"]["heldout"]
    gold_n = sum(selection["verified_sources"].values())
    cfgs = list(product(MODELS, PROMPTS))
    lookup = {
        (row["metric"], row["model"], row["prompt"], row["variant"], row["subset"]): row
        for row in all_metrics
    }

    def metric(name, model, prompt, style, subset="all"):
        return lookup[name, model, prompt, style, subset]

    def cell(name, model, prompt, style, subset="all"):
        return pct(metric(name, model, prompt, style, subset)["estimate"])

    def config(model, prompt):
        return f"{SIZES[model]} {prompt}"

    flip_table = table(
        ["Judge and prompt", "Verb.", "Conf.", "Fmt."],
        [[config(m, p)] + [cell("flip_rate", m, p, style) for style in EDITS] for m, p in cfgs],
        "Preference flip rate in percent. Each cell compares 200 same-order verdicts, "
        "clustered into 100 source questions.", "flips",
    )
    flip_scope = [float(metric("flip_rate", m, p, style)["estimate"])
                  for m, p, style in product(MODELS, PROMPTS, EDITS)]
    rubric_table = table(
        ["Qwen3", "Verb.", "Conf.", "Fmt."],
        [[SIZES[m]] + [signed(metric("rubric_flip_effect", m, "rubric_minus_brief", v)["estimate"])
                       for v in EDITS] for m in MODELS],
        "Rubric minus brief flip rates in percentage points on the same questions. "
        "Negative values indicate fewer flips under the rubric.", "rubric",
    )
    stability_table = table(
        ["Judge and prompt", *COMPACT_STYLES, "Repeat"],
        [[config(m, p)] + [cell("order_consistency", m, p, style) for style in STYLES]
         + [cell("repeat_consistency", m, p, "original")] for m, p in cfgs],
        "Order consistency by answer version and unchanged-repeat consistency, in percent. "
        "Order consistency uses 100 questions per cell. Repeat consistency uses 200 judgments.",
        "orders",
    )
    ties_table = table(
        ["Judge and prompt", *COMPACT_STYLES],
        [[config(m, p)] + [cell("tie_rate", m, p, style) for style in STYLES] for m, p in cfgs],
        "Tie rates in percent, based on 200 judgments per cell.", "ties",
    )
    efficiency_table = table(
        ["Qwen3", "P50 ms", "P95 ms", "Peak GiB"],
        [[SIZES[t["model"]], f'{float(t["median_latency_ms"]):.1f}',
          f'{float(t["p95_latency_ms"]):.1f}', f'{float(t["peak_inference_allocated_gib"]):.2f}']
         for t in timing if t["variant"] == "all"],
        "Warm batch-one forward-pass latency and peak PyTorch CUDA memory allocated during "
        "inference. Timing uses 1,600 original or edited judgments per model.", "efficiency",
    )
    zero_repeats = all(float(metric("baseline_flip", m, p, "original")["estimate"]) == 0 for m, p in cfgs)
    if zero_repeats:
        repeated = "All 1,200 unchanged repeats agreed with their original verdict, so the measured baseline flip rate was 0\\%."
    else:
        repeats = [metric("baseline_flip", m, p, "original") for m, p in cfgs]
        changes = sum(int(row["numerator"]) for row in repeats)
        repeated = f"{changes} of 1,200 unchanged repeats changed verdict."
    if gold_n:
        accuracy = (
            f"Automated verification identified {gold_n} held-out questions with a reference "
            f"preference. Each accuracy estimate therefore uses {2 * gold_n} verdicts across "
            "both orders. This small subset is descriptive."
        )
        accuracy_table = table(
            ["Judge and prompt", *COMPACT_STYLES],
            [[config(m, p)] + [cell("reference_accuracy", m, p, v, "verified") for v in STYLES]
             for m, p in cfgs],
            "Reference accuracy in percent on the automatically verified subset. "
            "Ties count as incorrect.", "accuracy",
        )
        delta_table = table(
            ["Judge and prompt", "Verb.", "Conf.", "Fmt."],
            [[config(m, p)] + [
                signed(metric("reference_accuracy_change", m, p, v, "verified")["estimate"])
                for v in EDITS] for m, p in cfgs],
            "Edited minus original reference accuracy in percentage points.", "accuracy-delta",
        )
    else:
        coverage = selection["reference_status"]
        accuracy = (
            "The automated protocol yielded zero held-out pairs with a verified preference. "
            f"Both extracted conclusions matched the source for {coverage.get('both_match', 0)} "
            f"pairs. At least one candidate conclusion was ambiguous for "
            f"{coverage.get('ambiguous_candidate_conclusion', 0)} pairs, and "
            f"{coverage.get('unsupported_source_answer', 0)} source answers used unsupported forms. "
            "Reference accuracy and its change are undefined in this sample. "
            "The other measures use all 100 held-out questions."
        )
        accuracy_table = delta_table = ""
    selected_reference = (
        f"Reference accuracy was restricted to {gold_n} verified held-out questions."
        if gold_n else "Reference accuracy was undefined because the automated checks verified zero stronger answers."
    )
    released_n = selection["verbosity_provenance"].get("CALM verbatim repetition", 0)
    mode = "4-bit NF4 with BF16 computation" if summary["precision"] == "nf4" else "BF16"
    compute = (
        f"The runs used {mode} on a single NVIDIA RTX PRO 6000 Blackwell Max-Q "
        f"Workstation Edition (96 GB). Development and held-out stages together charged "
        f"{summary['gpu_hours_charged']:.2f} GPU hours, including model loads and retries. "
        "After six warm-up judgments per model session, timing covers the forward pass and "
        "the return of the three scores to the CPU. Peak memory is the PyTorch allocation "
        "high-water mark after the model loads."
    )
    abstract = (
        "We test whether small Qwen3 models change their pairwise judgments after "
        "adding redundant text, confidence language, or Markdown emphasis. "
        "On 100 held-out CALM questions, three models evaluated the original and three "
        "edited versions in both answer orders with a brief prompt and a content rubric. "
        f"Across configurations, style edits changed {pct(min(flip_scope))}--{pct(max(flip_scope))}\\% "
        f"of judgments. {repeated} {selected_reference} We report paired "
        "rubric effects, order consistency, ties, and local inference costs."
    )
    results_tex = "\n\n".join([
        r"\paragraph{Coverage and repeatability.}",
        "The held-out ledger contains 4,800 original and edited judgments plus 1,200 "
        "unchanged repeats. The separate development run contains 1,200 judgments. "
        f"{released_n} of the 100 held-out questions use a released CALM verbosity edit. "
        f"The remaining {100 - released_n} use scripted duplication. "
        + repeated + " Under deterministic decoding, repeat consistency measures run-to-run "
        "reproducibility for identical inputs.",
        r"\paragraph{Style sensitivity and rubric effects.}",
        r"Table~\ref{tab:flips} gives the full comparison by model and prompt. Figure~\ref{fig:flips} "
        r"adds question-clustered intervals. Table~\ref{tab:rubric} gives the paired rubric effect for "
        r"each edit. Tables~\ref{tab:orders} and~\ref{tab:ties} show order consistency and ties "
        "across the same configurations.",
        r"\begin{figure*}[t]" "\n" r"\centering" "\n"
        r"\includegraphics[width=.94\textwidth]{generated/flip_rates.pdf}" "\n"
        r"\caption{Same-order preference flips by model, prompt, and edit. Whiskers are 95\% "
        r"question-clustered bootstrap intervals. Each point uses the same 100 held-out questions.}"
        "\n" r"\label{fig:flips}" "\n" r"\end{figure*}",
        flip_table, rubric_table, stability_table, ties_table,
        r"\paragraph{Reference accuracy.}", accuracy, accuracy_table, delta_table,
        r"\paragraph{Inference cost.}", compute, efficiency_table,
    ])
    prompt_text = "\n\n".join(rf"\paragraph{{{name.capitalize()} prompt.}} " + latex(value)
                              for name, value in PROMPTS.items())
    examples_table = table(
        ["Edit and pair", "Changed", "To target"],
        [[f"{item['variant'].capitalize()} {item['row']['pair_id']}", str(len(item["changes"])),
          str(sum(change["after"] == "target" for change in item["changes"]))] for item in examples],
        "Example cases selected by affected models and configurations. Each case has 12 matched "
        "comparisons across the three models, two prompts, and two orders.", "examples",
    ) if examples else ""
    appendix = "\n\n".join([
        prompt_text,
        r"\paragraph{Answer layout.} The user message labels the question, Answer A, and Answer B "
        r"on separate lines and ends in \texttt{Verdict:}. Inference uses Qwen's non-thinking "
        "chat template. Each example's source question ID, answer order, token-input hash, "
        "decision scores, and outcome are included in the project artifacts.",
        r"\paragraph{Data snapshot.} The complete selection and exclusion counts are stored "
        r"in \texttt{selection.json}. The held-out manifest's first 12 SHA-256 characters are "
        rf"\texttt{{{selection['manifest_sha256'][:12]}}}. The source scan uses only the "
        "factual portion of the CALM release. The report template is ACL revision "
        rf"\texttt{{{ACL[:12]}}}.",
        r"\paragraph{Example selection.} For each edit we rank examples by verified harmful "
        "changes across models when available, then by the number of affected models and "
        "configurations, favoring movement toward the edited candidate. We take one distinct "
        "question per edit. These cases illustrate behavior from the full measured sample.",
        examples_table,
    ])
    conclusion = (
        "The same underlying answer can receive a different verdict after a change in presentation. "
        "The size of the effect depends on the edit, model, and prompt. The paired design "
        "estimates presentation sensitivity independently of baseline repeat variability. "
        "Agreement across answer orders and accuracy against verified preferences are "
        "additional requirements for using compact judges in evaluation."
    )
    dest = report / "generated"
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("abstract", abstract), ("results", results_tex), ("appendix", appendix), ("conclusion", conclusion),
    ):
        (dest / f"{name}.tex").write_text(content + "\n", encoding="utf-8")
    shutil.copyfile(results / "figures" / "flip_rates.pdf", dest / "flip_rates.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--report", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    generate(args.results, args.report)


if __name__ == "__main__":
    main()
