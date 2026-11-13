"""Adversarial core regression tests; offline and independent of API services."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("m71_round2_tests", HERE / "m7_1_unified.py")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class CoreRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = M.build_model(HERE / "data")

    def test_public_route_accepts_aliases_and_ignores_zero_components(self):
        expected = M.m4_choose_model(["CY", "CR", "BK"], [40, 9, 1])
        self.assertEqual(expected, M.m4_choose_model(["CR", "CY", "CY", "BK", "UB"], [9, 20, 20, 1, 0]))
        for names, parts in ((["CY", "CR"], [1]), (["CY", "CR"], [-1, 2]), ([""], [1]), ([None], [1])):
            with self.subTest(names=names, parts=parts), self.assertRaises(ValueError):
                M.m4_choose_model(names, parts)

    def test_m4_accepts_legacy_labels_through_api_and_cli(self):
        examples = [
            (["Light Wisteria", "UB"], [1, 1], "baseline"),
            (["CY", "Permanent Red", "BK"], [40, 9, 1], "baseline"),
            (["Killarney", "Contessa", "Bon Jour"], [1, 1, 1], "dualgate"),
            (["Ochre", "Umber"], [1, 1], "dualgate"),
        ]
        for names, parts, expected in examples:
            with self.subTest(names=names):
                self.assertEqual(M.m4_choose_model(names, parts)[0], expected)
                with redirect_stdout(io.StringIO()) as output:
                    M.cmd_m4_route(SimpleNamespace(pigments=",".join(names), parts=",".join(map(str, parts))))
                self.assertEqual(json.loads(output.getvalue())["chosen_model"], expected)
        with self.assertRaises(ValueError):
            self.model.predict_recipe(["Light Wisteria", "UB"], [1, 1])

    def test_m4_route_cutoffs_consistent_across_scales(self):
        examples = [
            (["CY", "CR", "BK"], [107, 15, 3], "baseline"),    # exactly 12% red
            (["CY", "CR", "BK"], [163, 34, 3], "dualgate"),   # exactly 1.5% black
            (["CY", "CR", "BK"], [70, 28, 2], "baseline"),    # exactly 70% yellow
            (["CY", "CR", "BK"], [33, 6, 1], "baseline"),     # exactly 2.5% black
            (["QM", "UB"], [3, 7], "baseline"),               # exactly 30% magenta
            (["QM", "UB"], [2.99999, 7], "dualgate"),
        ]
        for names, parts, expected in examples:
            for inputs in (parts, [p * 0.3 for p in parts], M.normalize(parts)):
                with self.subTest(names=names, parts=inputs):
                    self.assertEqual(M.m4_choose_model(names, inputs)[0], expected)

    def test_public_pair_entrypoint_matches_recipe_and_validates_inputs(self):
        expected, _ = self.model.predict_recipe(["CY", "UB"], [1, 1])
        for a, b, wa, wb in (("CY", "UB", 1, 1), ("ub", "cadmium yellow light", 7, 7)):
            actual, meta = self.model.predict_pair(a, b, wa, wb)
            self.assertEqual(actual, expected)
            self.assertTrue(meta["is_observed"])
        pure, meta = self.model.predict_pair("CY", "CY", 2, 1)
        self.assertEqual(pure, self.model.palette[M.ABBR["CY"]])
        self.assertFalse(meta["is_observed"])
        for a, b, wa, wb in (("CY", "UB", -1, 2), ("CY", "UB", 0, 0), ("CY", "UB", np.nan, 1), ("CY", "bogus", 1, 0)):
            with self.subTest(a=a, b=b, wa=wa, wb=wb), self.assertRaises(ValueError):
                self.model.predict_pair(a, b, wa, wb)

    def test_exact_capture_and_endpoint_provenance_are_distinct(self):
        _, captured = self.model.predict_pair("CY", "UB", 1, 1)
        _, nearby = self.model.predict_pair("CY", "UB", 1, 1 + 1e-8)
        _, endpoint = self.model.predict_pair("CY", "UB", 1, 0)
        self.assertEqual(captured["pair_confidence"], "pair_exact_observed")
        self.assertFalse(nearby["is_observed"])
        self.assertNotEqual(nearby["pair_confidence"], "pair_exact_observed")
        self.assertEqual(endpoint["pair_confidence"], "pair_endpoint_anchor")
        self.assertFalse(endpoint["is_observed"])

    def test_tiny_extra_components_have_correct_pure_limit(self):
        pure = self.model.palette[M.ABBR["CY"]]
        for parts in ([1, 1e-6, 1e-6], [1, 1e-12, 1e-12], [1, 1e-20, 1e-20], [1e308, 1e-308, 1e-308]):
            with self.subTest(parts=parts):
                actual, meta = self.model.predict_recipe(["CY", "QM", "UB"], parts)
                self.assertEqual(actual, pure)
                self.assertFalse(meta["is_observed"])
        for parts in ([1e308, 1e308], [1e-308, 1e-308]):
            actual, _ = self.model.predict_pair("CY", "UB", *parts)
            expected, _ = self.model.predict_pair("CY", "UB", 1, 1)
            self.assertEqual(actual, expected)

    def test_text_parsing_rejects_missing_fields_and_nonfinite_values(self):
        self.assertEqual(M.parse_number_list("1:2/3|4,5"), [1, 2, 3, 4, 5])
        self.assertEqual(M.parse_parts_text("1,2,3"), [1, 2, 3])
        self.assertEqual(M.parse_list_text("CY | UB"), ["CY", "UB"])
        for text in ("1,,2", "1/", "NaN", "inf", "1:bad"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                M.parse_number_list(text)
        for text in ("CY||UB", "CY,", None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                M.parse_list_text(text)

    def test_hex_vectors_and_metric_inputs_are_validated(self):
        self.assertEqual(M.clean_hex("a3f"), "#AA33FF")
        for color in ("#GGGGGG", "#12", "#00000000", None, np.nan):
            with self.subTest(color=color), self.assertRaises(ValueError):
                M.clean_hex(color)
        for vector in ([1, 2], [1, 2, 3, 4], [0, np.nan, 0], [0, np.inf, 0]):
            for fn in (M.linear_rgb_to_hex, M.linear_rgb_to_xyz, M.xyz_to_lab, M.lab_to_xyz, M.linear_rgb_to_oklab, M.oklab_to_linear_rgb):
                with self.subTest(vector=vector, fn=fn.__name__), self.assertRaises(ValueError):
                    fn(vector)
        for delta in (-1, np.nan, np.inf, True):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                M.match_pct(delta)
        for vector in ([50, np.inf, 0], [50, 0], [1e300, 1e300, 1e300]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                M.ciede2000(vector, [50, 0, 0])
        self.assertEqual(M.match_pct(0), 100)
        self.assertEqual(M.match_pct(120), 0)
        self.assertEqual(M.ciede2000([50, 0, 0], [50, 0, 0]), 0)

    def test_invalid_grid_controls_fail_before_enumeration(self):
        for kwargs in ({"max_colors": 0}, {"max_colors": 9}, {"max_colors": True}, {"max_colors": 2.0}, {"total_parts": 0}, {"total_parts": -1}, {"total_parts": 6.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                M.generate_candidates(**kwargs)
        for kwargs in ({"top_n": 0}, {"top_n": -1}, {"top_n": 2.0}, {"top_n": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                M.m7_unmix(self.model, "#556B3A", **kwargs)
        with self.assertRaisesRegex(ValueError, "Reduce"):
            M.generate_candidates(max_colors=8, total_parts=100)
        for args in ((0, 2), (3, 0), (-1, 2), (True, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                list(M.compositions(*args))
        self.assertEqual(list(M.compositions(2, 3)), [])

    def test_restricted_palette_search_contains_only_profile_members(self):
        restricted = M.MeasuredPairwiseModel({"CY": "#FEE100", "UB": "#19123F"}, self.model.observations)
        self.assertGreater(restricted.excluded_observations, 0)
        top = M.m7_unmix(restricted, "#226A34", max_colors=4, total_parts=6, top_n=20)
        self.assertEqual(len(top), 7)
        self.assertEqual(top.iloc[0]["predicted_dE00"], 0)

    def test_data_validation_canonicalizes_before_cardinality_and_rejects_conflicts(self):
        palette = {"CY": "#FEE100", "UB": "#19123F"}
        one = {"trial_id": "T1", "pigments_full": "CY|CY|UB", "parts_text": "1/1/2", "trycolors_result_hex": "#226A34"}
        duplicate = {"trial_id": "T2", "pigments_full": "CY|UB|BK", "parts_text": "3/3/0", "trycolors_result_hex": "#226A34"}
        model = M.MeasuredPairwiseModel(palette, pd.DataFrame([one, duplicate]))
        self.assertEqual(len(model.nary_anchors), 0)
        self.assertEqual(model.predict_recipe(["CY", "UB"], [1, 1])[0], "#226A34")
        loo = model.leave_one_out_captured_pairs()
        self.assertEqual(loo["heldout_recipe_group_size"].tolist(), [2, 2])
        self.assertNotIn("pair_exact_observed", loo["confidence_tier"].tolist())
        bad = dict(duplicate, trycolors_result_hex="#000000")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            M.MeasuredPairwiseModel(palette, pd.DataFrame([one, bad]))
        with self.assertRaisesRegex(ValueError, "Missing observation columns"):
            M.MeasuredPairwiseModel(palette, pd.DataFrame([{"wrong": 1}]))
        with self.assertRaisesRegex(ValueError, "Pure-pigment"):
            M.MeasuredPairwiseModel(palette, pd.DataFrame([dict(one, pigments_full="CY|UB", parts_text="1/0")]))

    def test_profile_files_and_curve_geometry_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(HERE / "data", target / "data")
            (target / "data" / "trycolors_ui_additional_binary_captures.csv").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                M.build_model(target / "data")
        curve = self.model.curves[(M.ABBR["CY"], M.ABBR["UB"])]
        for value in (-0.1, 1.1, np.nan, np.inf):
            for fn in (curve.predict, curve.nearest_spacing, curve.has_exact_ratio):
                with self.subTest(value=value, fn=fn.__name__), self.assertRaises(ValueError):
                    fn(value)
        with self.assertRaises(ValueError):
            M.PairCurve(curve.first, curve.second, np.array([0, 0.5, 0.5, 1]), curve.hexes[:4], curve.oklab[:4], curve.source_ids[:4])
        with self.assertRaisesRegex(ValueError, "canonical order"):
            M.PairCurve(curve.second, curve.first, np.array([0, 1]), [curve.hexes[-1], curve.hexes[0]], curve.oklab[[0, -1]], ["endpoint:UB", "endpoint:CY"])

    def test_close_pair_capture_nodes_are_not_silently_rounded_away(self):
        palette = {"CY": "#FEE100", "UB": "#19123F"}
        rows = [
            {"trial_id": "A", "pigments_full": "CY|UB", "parts_text": "1/1", "trycolors_result_hex": "#123456"},
            {"trial_id": "B", "pigments_full": "CY|UB", "parts_text": "1/1.0000000000004", "trycolors_result_hex": "#ABCDEF"},
            {"trial_id": "C", "pigments_full": "CY|UB", "parts_text": "1/1e-14", "trycolors_result_hex": "#EEEEEE"},
        ]
        model = M.MeasuredPairwiseModel(palette, pd.DataFrame(rows))
        self.assertEqual(model.predict_pair("CY", "UB", 1, 1)[0], "#123456")
        self.assertEqual(model.predict_pair("CY", "UB", 1, 1.0000000000004)[0], "#ABCDEF")
        self.assertEqual(model.predict_pair("CY", "UB", 1, 1e-14)[0], "#EEEEEE")
        self.assertEqual(model.predict_pair("CY", "UB", 1, 0)[0], "#FEE100")
        with self.assertRaisesRegex(ValueError, "indistinguishable"):
            M.MeasuredPairwiseModel(palette, pd.DataFrame([dict(rows[0], parts_text="1e308/1e-308")]))

    def test_batch_csv_preserves_hex_and_identifier_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / "targets.csv"
            path.write_text("target_id,target_hex,target_name\n0001,000012,NA\n0002,123456,\n", encoding="utf-8")
            args = SimpleNamespace(data_dir=HERE / "data", targets=path, outdir=base / "outputs", max_colors=4, total_parts=6, top_n=5)
            result = pd.DataFrame({"rank": [1], "predicted_match_pct": [100.0], "predicted_hex": ["#000000"]})
            with mock.patch.object(M, "build_model", return_value=self.model), mock.patch.object(M, "m7_unmix", return_value=result) as unmix:
                M.cmd_batch_unmix(args)
            self.assertEqual([call.args[1] for call in unmix.call_args_list], ["#000012", "#123456"])
            actual = pd.read_csv(args.outdir / "m7_1_unmix_top1_capture_sheet.csv", dtype=str, keep_default_na=False)
            self.assertEqual(actual["target_id"].tolist(), ["0001", "0002"])
            self.assertEqual(actual["target_name"].tolist(), ["NA", ""])

    def test_profile_csv_preserves_capture_ids_colors_and_source_text(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            shutil.copytree(HERE / "data", data)
            path = data / "trycolors_ui_nary_anchor_captures_h24_rescue.csv"
            records = pd.read_csv(path, dtype=str, keep_default_na=False)
            records["trial_id"] = [f"{i:04d}" for i in range(1, len(records) + 1)]
            records["trycolors_result_hex"] = "123456"
            records["source"] = "NA"
            records["notes"] = ""
            records.to_csv(path, index=False)
            model = M.build_model(data)
            predicted, meta = model.predict_recipe(["CY", "IY", "UB"], [4, 1, 1])
            self.assertEqual(predicted, "#123456")
            self.assertEqual(meta["anchor_trial_id"], "0001")
            self.assertEqual(meta["anchor_source"], "NA")
            self.assertEqual(meta["anchor_notes"], "")

    def test_batch_validates_all_targets_before_search_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index, frame in enumerate((pd.DataFrame({"target_hex": ["#123456", "#INVALID"]}), pd.DataFrame(columns=["target_hex"]), pd.DataFrame({"wrong": [1]}))):
                path = base / f"targets{index}.csv"
                frame.to_csv(path, index=False)
                outdir = base / f"output{index}"
                args = SimpleNamespace(data_dir=HERE / "data", targets=path, outdir=outdir, max_colors=4, total_parts=6, top_n=5)
                with self.subTest(index=index), mock.patch.object(M, "build_model", return_value=self.model), mock.patch.object(M, "m7_unmix") as unmix:
                    with self.assertRaises(ValueError):
                        M.cmd_batch_unmix(args)
                    unmix.assert_not_called()
                    self.assertFalse(outdir.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
