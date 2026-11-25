from fractions import Fraction
import sqlite3
import unittest

import numpy as np

from style_judges.analysis import collect_statistics
from style_judges.data import (
    apply_edits, choice_conclusion, numeric_conclusion, reference_for, validate_edits,
)
from style_judges.inference import SCHEMA, case_keys, ledger_remaining, row_key, rows_for
from style_judges.protocol import ORDERS, PROMPTS, VARIANTS, verdict_identity
from style_judges.sources import MODELS


class ExperimentTests(unittest.TestCase):
    def test_intermediate_total_is_not_a_final_answer(self):
        first = "The initial total is 24. Thus, 12 penguins remain."
        second = "The answer is 12."
        winner, status, evidence = reference_for("GSM8K", {"gold": "12"}, first, second)
        self.assertIsNone(winner)
        self.assertEqual(status, "both_match")
        self.assertEqual(evidence, ["12", "12"])
        self.assertIsNone(numeric_conclusion("The initial total is 24. That is the amount removed."))

    def test_scalar_conclusions_and_explicit_choices(self):
        self.assertEqual(numeric_conclusion(r"The answer is \boxed{\frac{1}{4}}."), Fraction(1, 4))
        self.assertEqual(reference_for("GSM8K", {"gold": "12"},
                                       "The final answer is 12.", "The final answer is 24.")[0],
                         "answer1")
        self.assertEqual(choice_conclusion("No, the material won't conduct electricity.", ["yes", "no"]), 1)
        self.assertIsNone(choice_conclusion("Yes. But the correct answer is no.", ["yes", "no"]))

    def test_style_round_trip(self):
        source = r"The value of \(x\) is 3. The original qualification still applies."
        variants, evidence = apply_edits(source, "A rewrite that cannot preserve the whole original.")
        validate_edits({"variants": variants, "edit_evidence": evidence})
        self.assertEqual(variants["verbosity"], source + "\n\n" + source)
        self.assertEqual(variants["confidence"], "Certainly.\n\n" + source)
        self.assertIn(r"\(x\)", variants["formatting"])

    def test_answer_order(self):
        self.assertEqual(verdict_identity("A", "fixed_first"), "fixed")
        self.assertEqual(verdict_identity("A", "target_first"), "target")
        self.assertEqual(verdict_identity("B", "fixed_first"), "target")
        self.assertEqual(verdict_identity("B", "target_first"), "fixed")
        self.assertTrue(all(verdict_identity("T", order) == "tie" for order in ORDERS))

    def test_persisted_key_prevents_duplicate_work(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.executescript(SCHEMA)
        row = ("development", "nf4", "hash", "Qwen3-0.6B", "revision", "pair", "MATH",
               "original", "fixed_first", "brief", "sha", 110, "A", "fixed", 3., 2., 1., 1.)
        db.execute("INSERT INTO judgments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        with self.assertRaises(sqlite3.IntegrityError):
            db.execute("INSERT INTO judgments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        saved = rows_for(db, "development", "nf4", "hash", "Qwen3-0.6B")
        remaining = case_keys([{"pair_id": "pair"}]) - {row_key(record) for record in saved}
        self.assertEqual(len(remaining), 19)
        db.close()

    def test_gpu_budget(self):
        seconds = ledger_remaining([{"split": "development", "elapsed_seconds": 3 * 3600}], "heldout")
        self.assertEqual(seconds, 21 * 3600)
        exhausted = [{"split": "development", "elapsed_seconds": 5 * 3600},
                     {"split": "heldout", "elapsed_seconds": 19 * 3600}]
        self.assertEqual(ledger_remaining(exhausted, "heldout"), 0)

    def test_flip_and_accuracy_denominators(self):
        rows = [{"source": "GSM8K", "reference": "fixed"}, {"source": "MATH", "reference": None}]
        original = np.array([["fixed", "fixed"], ["target", "tie"]])
        changed = np.array([["target", "fixed"], ["target", "target"]])
        verdicts = {(model, prompt, variant): changed if variant == "confidence" else original
                    for model in MODELS for prompt in PROMPTS for variant in VARIANTS}
        metrics, _ = collect_statistics(rows, verdicts)
        stats = {(r["metric"], r["variant"], r["subset"]): r for r in metrics
                 if r["model"] == "Qwen3-0.6B" and r["prompt"] == "brief"}
        self.assertEqual(stats["flip_rate", "confidence", "all"]["numerator"], 2)
        self.assertEqual(stats["flip_rate", "confidence", "all"]["denominator"], 4)
        self.assertEqual(stats["reference_accuracy", "confidence", "verified"]["denominator"], 2)
        self.assertEqual(stats["reference_accuracy_change", "confidence", "verified"]["estimate"], -.5)
        self.assertEqual(stats["baseline_flip", "original", "all"]["estimate"], 0.)


if __name__ == "__main__":
    unittest.main()
