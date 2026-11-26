"""Generate the revised ACL report's measured tables, figure, and text."""

import argparse
from itertools import product
from pathlib import Path
import shutil

from style_judges.extension import EXTENSION_RESULTS
from style_judges.extension_analysis import RULES
from style_judges.extension_data import EDIT_STYLES, JUDGEBENCH
from style_judges.protocol import PROMPTS, read_json
from style_judges.report import SIZES, latex, load_csv, pct, signed, table
from style_judges.sources import CALM, MODELS, ROOT


def generate(results, report):
    summary = read_json(results / "analysis" / "summary.json")
    screening = read_json(results / "data" / "screening.json")
    assert summary["core_calls"] == (32 * summary["judgebench_pairs"] + 16 * summary["calm_pairs"]) * 3
    metrics = load_csv(results / "analysis" / "metrics.csv")
    lookup = {(row["cohort"], row["precision"], row["model"], row["prompt"], row["rule"],
               row["condition"], row["metric"]): row for row in metrics}

    def get(name, model, prompt, arm, rule="ordered", cohort="judgebench", precision="nf4"):
        return lookup[cohort, precision, model, prompt, rule, arm, name]

    def estimate(*args, **kwargs):
        return get(*args, **kwargs)["estimate"]

    def cell(*args, **kwargs):
        return pct(estimate(*args, **kwargs))

    def interval(row):
        return f'{pct(row["ci_low"])} to {pct(row["ci_high"])}'

    primary = summary["primary_result"]
    model = "Qwen3-4B"
    wrong_arm = "verbosity_incorrect"
    brief_acc, edited_acc, right_acc, both_acc = [
        f"{100 * float(estimate('accuracy', model, 'brief', arm)):.2f}"
        for arm in ("original", wrong_arm, "verbosity_correct", "verbosity_both")
    ]
    wrong = get("correct_to_wrong", model, "brief", wrong_arm)
    tie = get("correct_to_tie", model, "brief", wrong_arm)
    replication = get("preference_flip", model, "brief", "verbosity", cohort="calm")
    loss = float(primary["estimate"])
    effect_wording = f"a {100 * abs(loss):.2f}-point {'loss' if loss >= 0 else 'gain'}"
    uncertainty = "The paired interval includes zero. " if primary["ci_low"] <= 0 <= primary["ci_high"] else ""

    abstract = (
        "A style edit can change a language-model judge's decision and its correctness. "
        f"We evaluate {summary['judgebench_pairs']} labeled pairs from JudgeBench and replicate "
        f"the original style experiment on {summary['calm_pairs']} fresh CALM questions. "
        "Each labeled pair receives mirrored edits that change the presentation of either "
        "the correct or the incorrect answer. For Qwen3-4B with a brief prompt, repeating the "
        f"incorrect answer changes accuracy from {brief_acc}\\% to {edited_acc}\\%, "
        f"{effect_wording}. {uncertainty}The edit switches "
        f"{wrong['raw_numerator']} of {wrong['raw_denominator']} originally correct decisions "
        "to the wrong answer. We compare this instability under a content rubric and pooling "
        "over both answer orders."
    )

    original_table = table(
        ["Judge and prompt", "Orig.", "Right", "Wrong", "Both"],
        [[f"{SIZES[m]} {p}"] + [cell("accuracy", m, p, arm)
                               for arm in ("original", "verbosity_correct", wrong_arm, "verbosity_both")]
         for m, p in product(MODELS, PROMPTS)],
        "JudgeBench accuracy in percent with ordered A/B/tie scoring. "
        "Right, wrong, and both specify which original answers were duplicated. "
        "Accuracy includes ties as errors.", "extension-verbosity",
    )
    rule_labels = {
        "ordered": "Single order",
        "pooled": "Order pool",
        "ordered_forced": "Single order, forced",
        "pooled_forced": "Order pool, forced",
    }
    mitigation_table = table(
        ["4B prompt and rule", "Orig.", "Wrong", "Cov."],
        [[f"{p.capitalize()} / {rule_labels[rule]}",
          cell("accuracy", model, p, "original", rule),
          cell("accuracy", model, p, wrong_arm, rule),
          cell("coverage", model, p, wrong_arm, rule)]
         for p, rule in product(PROMPTS, RULES)],
        "Accuracy and coverage in percent for Qwen3-4B. Wrong is accuracy with the incorrect "
        "answer duplicated. Cov. is the fraction choosing an answer in that condition. "
        "Forced choice scores the same stored logits with only two candidate labels.",
        "extension-mitigation",
    )
    calm_table = table(
        ["Judge and prompt", "Verb.", "Conf.", "Fmt."],
        [[f"{SIZES[m]} {p}"] + [cell("preference_flip", m, p, style, cohort="calm")
                               for style in EDIT_STYLES] for m, p in product(MODELS, PROMPTS)],
        "Same-order flip rates in percent on fresh CALM pairs using the original decoder.",
        "extension-calm",
    )
    rubric_effect = get("rubric_reduction_in_loss", model, "brief_minus_rubric", wrong_arm)
    pooling_effect = get("pooling_reduction_in_loss", model, "brief", wrong_arm, "pooled")
    precision_table = ""
    precision_text = ""
    if summary["precision_pairs"]:
        precision_text = (
            f"We reran all eight conditions for {summary['precision_pairs']} JudgeBench questions "
            "with BF16 weights. Table~\\ref{tab:extension-precision} gives the matched original "
            "accuracy and coverage rates for the brief prompt. The analysis files give all paired "
            "style shifts and both prompts. Input token hashes matched between precisions."
        )
        precision_table = table(
            ["Qwen3", "NF4 Acc.", "BF16 Acc.", "NF4 Cov.", "BF16 Cov."],
            [[SIZES[m]] + [cell(metric, m, "brief", "original", precision=prec)
                            for metric, prec in product(("accuracy", "coverage"), ("nf4_matched", "bf16"))]
             for m in MODELS],
            "Matched NF4 and BF16 results in percent on unedited labeled pairs with the brief prompt. "
            "Accuracy includes ties as errors. Coverage is the share selecting an answer.",
            "extension-precision",
        )
    else:
        precision_text = "The matched BF16 follow-up yielded no complete matrix within its allocation."

    paragraphs = [
        r"\paragraph{Primary contrast.}",
        f"Under the brief prompt, Qwen3-4B's accuracy is {brief_acc}\\% on the originals, "
        f"{right_acc}\\% when the correct answer is duplicated, {edited_acc}\\% when the incorrect "
        f"answer is duplicated, and {both_acc}\\% when both are duplicated. Incorrect-answer "
        f"duplication yields {effect_wording} (95\\% interval for loss {interval(primary)} points). "
        f"Among {wrong['raw_denominator']} originally correct ordered decisions, "
        f"{wrong['raw_numerator']} changed to the incorrect answer and {tie['raw_numerator']} "
        "changed to a tie. These are raw counts across pairs and orders. The estimates and "
        "intervals weight each source question equally.",
        r"Table~\ref{tab:extension-verbosity} compares duplication across all models and "
        r"prompts. Figure~\ref{fig:extension-style} shows the direction and size of all three "
        "edits when applied to either answer.",
        r"\begin{figure*}[t]" "\n" r"\centering" "\n"
        r"\includegraphics[width=.96\textwidth]{extension_generated/accuracy_changes.pdf}" "\n"
        r"\caption{Change in JudgeBench accuracy under the brief prompt and ordered A/B/tie "
        r"scoring. Each point compares an edited condition with its original on the same "
        r"questions. Whiskers are paired, question-clustered 95\% bootstrap intervals.}" "\n"
        r"\label{fig:extension-style}" "\n" r"\end{figure*}",
        original_table,
        r"\paragraph{Rubric and pooling.}",
        f"For 4B, the estimated reduction in incorrect-duplication loss from using the rubric "
        f"is {signed(rubric_effect['estimate'])} points (95\\% interval {interval(rubric_effect)}). "
        f"With the brief prompt, the estimated reduction from A/B/tie pooling is "
        f"{signed(pooling_effect['estimate'])} points (95\\% interval {interval(pooling_effect)}). "
        r"Positive reductions favor the intervention. Table~\ref{tab:extension-mitigation} shows "
        "the accuracy level and coverage for each 4B method.",
        mitigation_table,
        r"\paragraph{Fresh CALM results.}",
        r"The 4B brief-prompt duplication flip rate was 19.5\% on the original CALM sample "
        f"and is {pct(replication['estimate'])}\\% on fresh pairs (95\\% interval {interval(replication)}\\%). "
        r"Table~\ref{tab:extension-calm} reports all original-protocol comparisons on the fresh sample.",
        calm_table,
        r"\paragraph{Precision check.}", precision_text, precision_table,
        r"\paragraph{Local compute.}",
        f"The extension recorded {summary['pilot_calls']:,} development, "
        f"{summary['core_calls']:,} core held-out, and {summary['precision_calls']:,} "
        f"BF16 judgments. It charged {summary['extension_gpu_hours']:.2f} GPU hours locally, "
        f"or {summary['total_gpu_hours']:.2f} hours including the first experiment. Worker time "
        "includes model loading and warm-up. The runner checked remaining time before each model "
        "and enforced the combined six-hour cap.",
    ]
    if screened := summary.get("format_sensitivity_pairs"):
        screen = get("correctness_loss", model, "brief", wrong_arm,
                     cohort="judgebench_format_screen")
        paragraphs.extend([
            r"\paragraph{Exploratory format screen.}",
            f"A stricter screen for explicit output formats leaves {screened} labeled pairs. "
            f"The primary contrast on this subset is {signed(screen['estimate'])} points of "
            f"accuracy loss (95\\% interval {interval(screen)} points). We added this "
            "sensitivity check after inspecting cases. The primary sample stays fixed.",
        ])

    counts = ", ".join(f"{count} {domain}" for domain, count in sorted(summary["judgebench_domains"].items()))
    excluded = screening["judgebench_screening"]
    prompts = "\n\n".join(rf"\paragraph{{{label.capitalize()} prompt.}} {latex(prompt)}"
                          for label, prompt in PROMPTS.items())
    append = "\n\n".join([
        f"The held-out JudgeBench set contains {counts} pairs. Of 620 released pairs, "
        f"{excluded.get('coding', 0)} came from the coding category, "
        f"{excluded.get('over_token_limit', 0)} exceeded the prompt length limit, "
        f"and {excluded.get('invalid_edit', 0) + excluded.get('invalid_text', 0) + excluded.get('calm_overlap', 0)} "
        "failed another eligibility check.",
        rf"JudgeBench snapshot \texttt{{{JUDGEBENCH[:12]}}}. CALM snapshot \texttt{{{CALM[:12]}}}. "
        rf"Core manifest SHA-256 prefix \texttt{{{summary['core_manifest_sha256'][:12]}}}. "
        "The results files provide full hashes and retained pair IDs.",
        prompts,
    ])
    dest = report / "extension_generated"
    dest.mkdir(parents=True, exist_ok=True)
    facts = (rf"\newcommand{{\GoldPairs}}{{{summary['judgebench_pairs']}}}" "\n"
             rf"\newcommand{{\GoldQuestions}}{{{summary['judgebench_questions']}}}" "\n"
             rf"\newcommand{{\CalmPairs}}{{{summary['calm_pairs']}}}")
    for name, content in (("facts", facts), ("abstract", abstract),
                          ("results", "\n\n".join(p for p in paragraphs if p)), ("appendix", append)):
        (dest / f"{name}.tex").write_text(content + "\n", encoding="utf-8")
    shutil.copyfile(results / "figures" / "accuracy_changes.pdf", dest / "accuracy_changes.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=EXTENSION_RESULTS)
    parser.add_argument("--report", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    generate(args.results, args.report)


if __name__ == "__main__":
    main()
