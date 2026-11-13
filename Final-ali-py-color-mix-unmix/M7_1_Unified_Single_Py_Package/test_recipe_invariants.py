"""Focused regression tests for the recovered M7.1 recipe representation.

Run from any directory with: python /path/to/test_recipe_invariants.py
No network calls or training are performed.
"""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("m71_corrected", HERE / "m7_1_unified.py")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class RecipeInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = M.build_model(HERE / "data")

    def anchor_cases(self):
        for _, row in self.model.observations.iterrows():
            names = M.parse_pigments_full(row.get("pigments_full", ""))
            parts = M.parse_parts_text(row.get("parts_text", ""))
            if len(names) >= 3 and len(names) == len(parts):
                yield names, parts, M.clean_hex(row["trycolors_result_hex"])

    def test_every_measured_nary_anchor_is_scale_invariant(self):
        cases = list(self.anchor_cases())
        self.assertEqual(len(cases), 7)
        for names, parts, expected in cases:
            for factor in (1, 2, 10, 0.1):
                with self.subTest(names=names, parts=parts, factor=factor):
                    result, meta = self.model.predict_recipe(names, [p * factor for p in parts])
                    self.assertEqual(result, expected)
                    self.assertEqual(meta["confidence_tier"], "nary_exact_observed_anchor")

    def test_anchor_key_preserves_permutation_alias_and_scale(self):
        names, parts, _ = next(self.anchor_cases())
        aliases = [M.NAME_TO_ABBR[n] for n in reversed(names)]
        self.assertEqual(M.recipe_key(names, parts), M.recipe_key(aliases, [2 * p for p in reversed(parts)]))

    def test_public_nary_dictionary_keys_are_preserved(self):
        for names, parts, expected in self.anchor_cases():
            with self.subTest(names=names, parts=parts):
                self.assertEqual(self.model.nary_anchors[M.recipe_key(names, parts)]["hex"], expected)

    def test_duplicate_and_zero_parts_do_not_change_any_anchor(self):
        for names, parts, expected in self.anchor_cases():
            names2 = list(names) + [names[0], "TW"]
            parts2 = [parts[0] / 2] + parts[1:] + [parts[0] / 2, 0]
            with self.subTest(names=names):
                result, meta = self.model.predict_recipe(names2, parts2)
                self.assertEqual(result, expected)
                self.assertEqual(meta["confidence_tier"], "nary_exact_observed_anchor")

    def test_pure_pigment_and_binary_invariants(self):
        pure, _ = self.model.predict_recipe(["CY", "CY", "BK"], [1, 2, 0])
        self.assertEqual(pure, self.model.palette[M.ABBR["CY"]])
        a, ma = self.model.predict_recipe(["CY", "UB"], [1, 1])
        b, mb = self.model.predict_recipe(["UB", "CY", "CY", "BK"], [2, 1, 1, 0])
        self.assertEqual(a, b)
        self.assertEqual(ma, mb)

    def test_invalid_recipe_is_explicitly_rejected(self):
        bad = [([], []), (["CY", "UB"], [1]), (["CY"], [1, 2]),
               (["CY", "UB"], [-1, 2]), (["CY"], [float("nan")]),
               (["CY"], [float("inf")]), (["CY"], [0]),
               (["unknown pigment"], [1]), (["unknown pigment", "CY"], [0, 1])]
        for names, parts in bad:
            with self.subTest(names=names, parts=parts):
                with self.assertRaises(ValueError):
                    self.model.predict_recipe(names, parts)

    def test_recipe_is_scoped_to_this_models_palette(self):
        palette = dict(self.model.palette)
        del palette[M.ABBR["UB"]]
        restricted = M.MeasuredPairwiseModel(palette, self.model.observations)
        with self.assertRaisesRegex(ValueError, "measured palette"):
            restricted.predict_recipe(["UB"], [1])

    def test_candidate_grid_semantics_unchanged(self):
        candidates = M.generate_candidates(max_colors=4, total_parts=6)
        self.assertEqual(len(candidates), 1408)
        for _, row in candidates.iterrows():
            if len(row["pigment_names"]) == 1:
                self.assertEqual(row["parts"], [1])
            else:
                self.assertEqual(sum(row["parts"]), 6)
                self.assertTrue(all(isinstance(p, int) and p > 0 for p in row["parts"]))

    def test_normalize_checks_nonfinite_and_preserves_large_ratios(self):
        self.assertEqual(M.normalize([1e308, 1e308]), [0.5, 0.5])
        for parts in ([], [0, 0], [-1, 2], [float("nan")], [float("inf")]):
            with self.subTest(parts=parts):
                with self.assertRaises(ValueError):
                    M.normalize(parts)

    def test_nary_lookup_crosses_decimal_hash_rounding_boundary(self):
        row = {"trial_id": "new", "pigments_full": "CY|QM|UB", "parts_text": "3/4/36", "trycolors_result_hex": "#123456"}
        model = M.MeasuredPairwiseModel(self.model.palette, pd.DataFrame([row]))
        # Multiplication in ordinary Python introduces harmless binary-float noise.
        noisy = [value * 0.3 for value in [3, 4, 36]]
        self.assertNotEqual(M.recipe_key(["CY", "QM", "UB"], [3, 4, 36]), M.recipe_key(["CY", "QM", "UB"], noisy))
        for names, parts in ((["CY", "QM", "UB"], noisy), (["CY", "QM", "UB", "CY"], [0.3, 1.2, noisy[2], 0.6])):
            with self.subTest(names=names, parts=parts):
                actual, meta = model.predict_recipe(names, parts)
                self.assertEqual(actual, "#123456")
                self.assertEqual(meta["confidence_tier"], "nary_exact_observed_anchor")

    def test_conflicting_equivalent_captures_rejected_across_hash_bins(self):
        rows = [
            {"trial_id": "A", "pigments_full": "CY|UB", "parts_text": "18/133", "trycolors_result_hex": "#123456"},
            {"trial_id": "B", "pigments_full": "CY|UB", "parts_text": "5.3999999999999995/39.9", "trycolors_result_hex": "#ABCDEF"},
        ]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            M.MeasuredPairwiseModel(self.model.palette, pd.DataFrame(rows))

    def test_grouped_holdout_removes_float_noise_variants(self):
        rows = [
            {"trial_id": "A", "pigments_full": "CY|UB", "parts_text": "18/133", "trycolors_result_hex": "#123456"},
            {"trial_id": "B", "pigments_full": "CY|UB", "parts_text": "5.3999999999999995/39.9", "trycolors_result_hex": "#123456"},
        ]
        model = M.MeasuredPairwiseModel(self.model.palette, pd.DataFrame(rows))
        self.assertEqual(model.observations["_recipe_group"].nunique(), 1)
        heldout = model.leave_one_out_captured_pairs()
        self.assertEqual(heldout["heldout_recipe_group_size"].tolist(), [2, 2])
        self.assertEqual(heldout["confidence_tier"].tolist(), ["pair_unmeasured_interpolation"] * 2)

    def test_distinct_binary_captures_in_one_hash_bin_are_preserved(self):
        names = ["CY", "UB"]
        self.assertEqual(M.recipe_key(names, [1, 1]), M.recipe_key(names, [1, 1.00000000000001]))
        rows = [
            {"trial_id": "A", "pigments_full": "CY|UB", "parts_text": "1/1", "trycolors_result_hex": "#123456"},
            {"trial_id": "B", "pigments_full": "CY|UB", "parts_text": "1/1.00000000000001", "trycolors_result_hex": "#ABCDEF"},
        ]
        model = M.MeasuredPairwiseModel(self.model.palette, pd.DataFrame(rows))
        self.assertEqual(model.observations["_recipe_group"].nunique(), 2)
        self.assertEqual(model.predict_recipe(names, [1, 1])[0], "#123456")
        self.assertEqual(model.predict_recipe(names, [1, 1.00000000000001])[0], "#ABCDEF")

    def test_distinct_nary_captures_in_one_hash_bin_are_preserved(self):
        names = ["CY", "QM", "UB"]
        self.assertEqual(M.recipe_key(names, [4, 1, 1]), M.recipe_key(names, [4, 1.00000000000001, 1]))
        rows = [
            {"trial_id": "A", "pigments_full": "CY|QM|UB", "parts_text": "4/1/1", "trycolors_result_hex": "#123456"},
            {"trial_id": "B", "pigments_full": "CY|QM|UB", "parts_text": "4/1.00000000000001/1", "trycolors_result_hex": "#ABCDEF"},
        ]
        model = M.MeasuredPairwiseModel(self.model.palette, pd.DataFrame(rows))
        self.assertEqual(len(model.nary_anchors), 2)
        self.assertEqual(model.predict_recipe(names, [4, 1, 1])[0], "#123456")
        self.assertEqual(model.predict_recipe(names, [4, 1.00000000000001, 1])[0], "#ABCDEF")

    def test_clearing_nary_anchors_disables_capture_lookup(self):
        row = {"trial_id": "A", "pigments_full": "CY|QM|UB", "parts_text": "4/1/1", "trycolors_result_hex": "#123456"}
        model = M.MeasuredPairwiseModel(self.model.palette, pd.DataFrame([row]))
        model.nary_anchors.clear()
        actual, meta = model.predict_recipe(["CY", "QM", "UB"], [4, 1, 1])
        self.assertFalse(meta["is_observed"])
        self.assertNotEqual(actual, "#123456")


if __name__ == "__main__":
    unittest.main(verbosity=2)
