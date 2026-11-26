import unittest

import numpy as np

from style_judges.extension_analysis import EXPLICIT_FORMAT, QuestionBootstrap, decisions
from style_judges.extension_data import (
    GOLD_ARMS, group_judgebench, mirrored_arms, ranked_groups, upstream_question_key,
)
from style_judges.inference import case_keys


class ExtensionTests(unittest.TestCase):
    def test_mirrored_edits_follow_the_gold_identity(self):
        a = "We computed this answer based on the problem and obtained twenty."
        b = "A different calculation produces fifteen as the final answer here."
        arms = mirrored_arms(a, b, "target")
        self.assertEqual(tuple(arms), GOLD_ARMS)
        self.assertEqual(arms["verbosity_correct"], [a, b + "\n\n" + b])
        self.assertEqual(arms["verbosity_incorrect"], [a + "\n\n" + a, b])
        self.assertEqual(arms["confidence_incorrect"], ["Certainly.\n\n" + a, b])
        self.assertEqual(len(case_keys([{"pair_id": "benchmark", "answer_variants": arms}])), 32)

    def test_missing_upstream_ids_do_not_merge_unrelated_questions(self):
        source = "livebench-math"
        a = upstream_question_key(source, None, "A question")
        same = upstream_question_key(source, None, "A  question")
        other = upstream_question_key(source, None, "Another question")
        self.assertEqual(a, same)
        self.assertNotEqual(a, other)
        rows = [
            {"source_id": a, "question": "A question"},
            {"source_id": "an alternate upstream id", "question": "A question"},
            {"source_id": other, "question": "Another question"},
        ]
        group_judgebench(rows)
        self.assertEqual(rows[0]["question_group"], rows[1]["question_group"])
        self.assertNotEqual(rows[0]["question_group"], rows[2]["question_group"])

    def test_group_rank_prefix_is_deterministic(self):
        rows = [{"question_group": f"question-{i}", "domain": "math" if i % 2 else "knowledge"}
                for i in range(20)]
        ranked = ranked_groups(rows, "unit-test", 20)
        self.assertEqual(ranked[:6], ranked_groups(list(reversed(rows)), "unit-test", 6))
        self.assertEqual(len(set(ranked)), 20)

    def test_pooling_maps_both_orders_and_keeps_ties_visible(self):
        # In both orders the first identity beats the second, while the tie token dominates.
        scores = np.array([[[3.0, 0.0, 8.0], [0.0, 3.0, 8.0]],
                           [[0.0, 3.0, -2.0], [3.0, 0.0, -2.0]]])
        ordered = np.array([[2, 2], [1, 1]])
        result = decisions(scores, ordered)
        np.testing.assert_array_equal(result["pooled"].ravel(), [2, 1])
        np.testing.assert_array_equal(result["pooled_forced"].ravel(), [0, 1])
        np.testing.assert_array_equal(result["ordered_forced"], [[0, 0], [1, 1]])

    def test_bootstrap_uses_question_weights_and_conditional_denominators(self):
        rows = [{"question_group": group, "domain": "math"} for group in ("one", "one", "two")]
        stats = QuestionBootstrap(rows, "judgebench", "nf4")
        stats.add("correctness_loss", "model", "brief", "ordered", "edited",
                  np.array([[1, 1], [1, 1], [0, 0]]))
        self.assertEqual(stats.metrics[-1]["estimate"], .5)
        self.assertEqual(stats.metrics[-1]["raw_numerator"], 4)
        self.assertEqual(stats.metrics[-1]["raw_denominator"], 6)
        stats.add("conditional_accuracy", "model", "brief", "ordered", "edited",
                  np.zeros((3, 2)), np.zeros((3, 2)))
        self.assertIsNone(stats.metrics[-1]["estimate"])
        self.assertEqual(stats.metrics[-1]["raw_denominator"], 0)

    def test_exploratory_format_screen_detects_explicit_bold_request(self):
        self.assertIsNotNone(EXPLICIT_FORMAT.search("Put your answer in **bold** as a single integer."))
        self.assertIsNone(EXPLICIT_FORMAT.search("Which answer is most likely? The heading is bold."))


if __name__ == "__main__":
    unittest.main()
