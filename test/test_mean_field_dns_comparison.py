"""Independent benchmark, matched inputs, comparison metrics, and plot layout."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import mixing_layer_dns as dns
import mixing_layer_mean_field as mf
from compare_mean_field_dns import dns_config, run_reference
from plot_mean_field_results import (comparison, field_coordinates, identical_initial_fields,
                                    load_fields, plot_dns_comparison, plot_dns_diagnostics, verify_dns_setup)
from postprocess_mean_field_run import postprocess


class DNSComparisonTests(unittest.TestCase):
    def test_maps_all_physical_parameters_without_boson_encoding_scales(self):
        cfg = mf.MeanFieldConfig(reynolds=80, peclet=90, velocity_scale=12, scalar_scale=8)
        actual = dns_config(cfg)
        self.assertEqual(actual.viscosity, 1/80)
        self.assertEqual(actual.scalar_diffusivity, 1/90)
        self.assertEqual(actual.reference_velocity, 1)
        verify_dns_setup(cfg, {"config": asdict(actual)})
        altered = asdict(actual)
        altered["reynolds"] = 50
        with self.assertRaisesRegex(ValueError, "reynolds"):
            verify_dns_setup(cfg, {"config": altered})

    def test_comparison_norms_and_rejects_incompatible_or_nonfinite_data(self):
        reference = {"times": np.array([0., .1]), "u": np.ones((2, 4, 4)),
                     "v": np.ones((2, 5, 4)), "vorticity": np.ones((2, 5, 4)),
                     "concentration": np.ones((2, 4, 4))}
        primary = {key: value.copy() if key == "times" else 2*value for key, value in reference.items()}
        rows = comparison(primary, reference)
        for row in rows:
            for field in ("velocity", "vorticity", "concentration"):
                self.assertAlmostEqual(row[field+"_relative_l2"], 1)
        self.assertFalse(identical_initial_fields(primary, reference))
        self.assertTrue(identical_initial_fields(reference, reference))
        for key, value in (("times", np.array([0.])), ("times", np.array([0., .2])),
                           ("v", np.ones((2, 4, 4))), ("u", np.full((2, 4, 4), np.nan))):
            bad = {**reference, key: value}
            with self.assertRaises(ValueError):
                comparison(primary, bad)

    def test_vertex_coordinates_include_the_periodic_endpoint(self):
        for boundary, rows in (("free-slip", 5), ("periodic", 4)):
            cfg = mf.MeanFieldConfig(n=4, boundary_y=boundary)
            values = np.arange(rows*4).reshape(rows, 4)
            x, y, field = field_coordinates(values, cfg, "vorticity")
            self.assertEqual(field.shape, (5, 5))
            self.assertEqual(x[-1], 1)
            self.assertEqual(y[-1], 1)
            np.testing.assert_array_equal(field[:, -1], field[:, 0])

    def test_reference_is_independent_and_bitwise_matches_initial_observables(self):
        cfg = mf.MeanFieldConfig(n=4, kh_amplitude=.2, secondary_amplitude=.03, final_time=.0175)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = mf.run_simulation(cfg, root/"mf")
            digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
            path = root/"dns.npz"
            validation = run_reference(source, path)
            self.assertEqual(digest_before, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertTrue(validation["validated"])
            self.assertEqual(validation["completed_steps"], cfg.steps)
            primary, reference = load_fields(source), load_fields(path)
            verify_dns_setup(cfg, reference)
            self.assertTrue(identical_initial_fields(primary, reference))
            np.testing.assert_array_equal(primary["times"], reference["times"])
            initial_error = comparison(primary, reference)[0]
            for key, value in initial_error.items():
                self.assertEqual(value, 0)
            expected = dns.advance_one_step_with_scalar(primary["u"][0], primary["v"][0],
                         primary["concentration"][0], cfg.dt, dns_config(cfg))
            for name, values in zip(("u", "v", "pressure", "concentration"), expected):
                if name != "pressure":
                    np.testing.assert_array_equal(reference[name][1], values)
            self.assertEqual(reference["run_metadata"]["mass_projection"], False)
            self.assertLess(validation["worst_step"]["scalar_mass_error"], 1e-12)
            for field in ("vorticity", "concentration"):
                destination = root/f"compare_{field}.png"
                plot_dns_comparison(primary, reference, cfg, field, destination)
                self.assertGreater(destination.stat().st_size, 1000)
            plot_dns_diagnostics(primary, reference, cfg, root/"diagnostics.png")
            self.assertGreater((root/"diagnostics.png").stat().st_size, 1000)

    def test_automatic_postprocessing_creates_validated_report_and_plots(self):
        cfg = mf.MeanFieldConfig(n=4, kh_amplitude=.2, secondary_amplitude=.03, final_time=.0175)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = mf.run_simulation(cfg, root/"mf")
            published = root/"published"
            summary = postprocess(source, published)
            self.assertTrue(summary["references"]["DNS"]["identical_initial_fields"])
            self.assertEqual(json.loads((published/"postprocess_status.json").read_text())["state"], "complete")
            self.assertIn("explicit single-site", (published/"RUN_REPORT.md").read_text())
            for name in ("mean_field_vorticity.png", "mean_field_concentration.png",
                         "mean_field_vs_dns_vorticity.png", "mean_field_vs_dns_concentration.png",
                         "mean_field_vs_dns_diagnostics.png"):
                self.assertGreater((published/name).stat().st_size, 1000)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             hashlib.sha256((published/"mean_field_snapshots.npz").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
