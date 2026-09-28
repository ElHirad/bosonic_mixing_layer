"""Four-stage DNS convergence and unclipped signed-species observations."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import mixing_layer_dns as dns
import mixing_layer_mean_field as mf
import reacting_dns
from compare_mean_field_dns import dns_config, matched_dns_integrator, run_reference
from plot_mean_field_results import load_fields, reaction_sign_comparison, verify_dns_setup


class ReactingRK4Tests(unittest.TestCase):
    def test_uniform_reaction_fourth_order_and_conservation(self):
        cfg = dns_config(mf.MeanFieldConfig(n=4, scalar_species=3, damkohler=100))
        errors = []
        for dt in (.002, .001, .0005):
            u, v = np.zeros((4, 4)), np.zeros((5, 4))
            c = np.stack([np.full((4, 4), .5), np.full((4, 4), .5), np.zeros((4, 4))])
            for _ in range(round(.02/dt)):
                u, v, _, c = reacting_dns.advance_one_step(u, v, c, dt, cfg, 100, method="rk4")
            errors.append(abs(c[0, 0, 0]-.25))
            np.testing.assert_allclose(c[0]+c[2], .5, atol=2e-15, rtol=0)
            np.testing.assert_allclose(c[1]+c[2], .5, atol=2e-15, rtol=0)
        for ratio in (errors[0]/errors[1], errors[1]/errors[2]):
            self.assertGreater(ratio, 13)
            self.assertLess(ratio, 20)

    def test_fourth_order_projected_velocity_and_scalar_diffusion_both_boundaries(self):
        n, end_time = 8, .04
        y = (np.arange(n)+.5)/n
        for boundary in ("periodic", "free-slip"):
            cfg = dns_config(mf.MeanFieldConfig(n=n, reynolds=1., peclet=1., boundary_y=boundary))
            wave = 2*np.pi if boundary == "periodic" else np.pi
            profile = np.repeat(np.cos(wave*y)[:, None], n, axis=1)
            eigenvalue = -4*n*n*np.sin(wave/(2*n))**2
            exact = np.exp(eigenvalue*end_time)*profile
            errors = []
            for dt in (.005, .0025, .00125):
                u = profile.copy()
                v = np.zeros((n if boundary == "periodic" else n+1, n))
                c = np.stack([.5+.1*profile, .5-.1*profile, np.zeros_like(profile)])
                for _ in range(round(end_time/dt)):
                    u, v, p, c = reacting_dns.advance_one_step(u, v, c, dt, cfg, 0, method="rk4")
                errors.append(np.linalg.norm(u-exact))
                np.testing.assert_allclose(c[0], .5+.1*u, atol=2e-15, rtol=0)
                np.testing.assert_allclose(c[0]+c[2]+c[1], 1., atol=2e-15, rtol=0)
                self.assertLess(abs(p.mean()), 1e-14)
                if boundary == "free-slip":
                    np.testing.assert_array_equal(v[[0, -1]], 0.)
            for ratio in (errors[0]/errors[1], errors[1]/errors[2]):
                self.assertGreater(ratio, 14)
                self.assertLess(ratio, 19)

    def test_exactly_four_coupled_rhs_evaluations_and_no_species_clipping(self):
        cfg = dns_config(mf.MeanFieldConfig(n=4))
        u, v = np.zeros((4, 4)), np.zeros((5, 4))
        c = np.stack([np.full((4, 4), -.01), np.full((4, 4), .4), np.full((4, 4), .02)])
        with patch.object(dns, "channel_momentum_rhs", wraps=dns.channel_momentum_rhs) as momentum, \
             patch.object(reacting_dns, "species_rhs", wraps=reacting_dns.species_rhs) as species:
            _, _, _, result = reacting_dns.advance_one_step(u, v, c, .001, cfg, 0, method="rk4")
        self.assertEqual(momentum.call_count, 4)
        self.assertEqual(species.call_count, 4)
        np.testing.assert_array_equal(result, c)

    def test_reactive_rk4_reference_rejects_midpoint_mislabelling(self):
        cfg = mf.MeanFieldConfig(n=4, scalar_species=3, damkohler=1, time_integrator="rk4")
        self.assertEqual(matched_dns_integrator(cfg), "rk4")
        reference = dict(config={**asdict(dns_config(cfg)), "scalar_species": 3,
                                 "damkohler": 1., "time_integrator": "rk4"})
        verify_dns_setup(cfg, reference)
        reference["config"]["time_integrator"] = "midpoint"
        with self.assertRaisesRegex(ValueError, "time_integrator"):
            verify_dns_setup(cfg, reference)
        self.assertEqual(matched_dns_integrator(mf.MeanFieldConfig()), "midpoint")


class ConcentrationSignTests(unittest.TestCase):
    def test_both_observers_count_signs_and_preserve_signed_amounts(self):
        cfg = mf.MeanFieldConfig(n=4, scalar_species=3, damkohler=1)
        c1 = np.array([-1., -1e-14, 0., 2.] * 4).reshape(4, 4)
        fields = dict(c1=c1, c2=1-c1, c3=np.zeros_like(c1))
        reference = mf.scalar_references(fields, cfg)
        species = np.stack([fields[f"c{i}"] for i in (1, 2, 3)])
        before = species.copy()
        first = mf.scalar_diagnostics(fields, cfg, reference)
        second = reacting_dns.conservation_diagnostics(species, reference)
        for row in (first, second):
            self.assertEqual(row["c1_positive_cells"], 4)
            self.assertEqual(row["c1_negative_cells"], 8)
            self.assertEqual(row["c1_zero_cells"], 4)
            self.assertEqual(row["c1_material_negative_cells"], 4)
            self.assertEqual(row["c1_positive_fraction"], .25)
            self.assertEqual(row["c1_negative_fraction"], .5)
            self.assertEqual(row["c1_positive_integral"], .5)
            self.assertAlmostEqual(row["c1_negative_integral"], -.2500000000000025)
            for name in ("c1", "c2", "c3"):
                self.assertAlmostEqual(row[name+"_positive_integral"]+row[name+"_negative_integral"],
                                       row[name+"_mass"])
        np.testing.assert_array_equal(species, before)

    def test_rk4_matching_and_signed_csv_include_every_step_and_initial_state(self):
        cfg = mf.MeanFieldConfig(n=4, scalar_species=3, damkohler=10,
            kh_amplitude=.02, secondary_amplitude=.003, dt=.0001, final_time=.0007,
            reynolds=100, peclet=100, time_integrator="rk4", pressure_tolerance=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = mf.run_simulation(cfg, root/"mf", checkpoint_interval=1)
            run_reference(source, root/"dns.npz")
            primary, reference = load_fields(source), load_fields(root/"dns.npz")
            self.assertEqual(reference["run_metadata"]["time_integrator"], "rk4")
            verify_dns_setup(cfg, reference)
            summary = reaction_sign_comparison([(primary, "MF", "-"), (reference, "DNS", "--")], cfg, root)
            self.assertTrue(summary["includes_initial_state"])
            with (root/"concentration_sign_history.csv").open() as file:
                records = list(csv.DictReader(file))
            self.assertEqual(len(records), 2*3*(cfg.steps+1))
            for method in ("MF", "DNS"):
                for name in ("c1", "c2", "c3"):
                    selected = [r for r in records if r["method"] == method and r["species"] == name]
                    self.assertEqual([int(r["step"]) for r in selected], list(range(cfg.steps+1)))
                    for row in selected:
                        self.assertAlmostEqual(float(row["positive_integral"])+float(row["negative_integral"]),
                                               float(row["mass"]), places=14)
            self.assertGreater((root/"concentration_signs.png").stat().st_size, 1000)
            with np.load(source, allow_pickle=False) as data:
                altered = {key: data[key].copy() for key in data.files}
            history = json.loads(str(altered["history_json"]))
            history[0]["c1_negative_cells"] += 1
            altered["history_json"] = np.array(json.dumps(history))
            mf.atomic_npz(root/"corrupt.npz", **altered)
            with self.assertRaisesRegex(ValueError, "saved scalar diagnostic"):
                mf.validate_results(root/"corrupt.npz")


if __name__ == "__main__":
    unittest.main()
