"""Reaction stoichiometry, explicit operators, independent DNS, and persistence."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import mixing_layer_mean_field as mf
import mixing_layer_dns as dns
import reacting_dns
from compare_mean_field_dns import dns_config, run_reference
from mean_field_bosons import LocalBosons
from mean_field_operators import CoherentGenerator, MeanFieldOperators
from plot_mean_field_results import comparison, identical_initial_fields, load_fields, verify_dns_setup
from postprocess_mean_field_run import postprocess


def config(**changes):
    return replace(mf.MeanFieldConfig(n=4, kh_amplitude=.2, secondary_amplitude=.03,
        final_time=.0175, scalar_species=3, damkohler=100), **changes)


def setup(cfg):
    b, ops = LocalBosons(cfg.boson_cutoff), MeanFieldOperators(cfg)
    states = b.coherent_states(ops.layout.pack(*mf.initial_amplitudes(cfg)))
    return b, ops, states


class ReactionTests(unittest.TestCase):
    def test_initial_complement_and_zero_product(self):
        cfg = config()
        b, ops, states = setup(cfg)
        fields = mf.read_fields(states, b, ops, cfg)
        np.testing.assert_allclose(fields["c2"], 1-fields["c1"], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(fields["c3"], 0)
        np.testing.assert_allclose(mf.scalar_references(fields, cfg), [.5, .5], atol=1e-15)
        self.assertEqual(ops.layout.size, 6*cfg.n**2+cfg.n)
        self.assertEqual(ops.layout.scalar_slice.stop-ops.layout.scalar_slice.start, 3*cfg.n**2)

    def test_predictor_matches_independent_reactive_stencils_and_invariants(self):
        rng = np.random.default_rng(92)
        for boundary in ("free-slip", "periodic"):
            cfg = config(boundary_y=boundary, scalar_scale=5, velocity_scale=3, pressure_scale=7)
            _, ops, _ = setup(cfg)
            u = rng.normal(size=ops.layout.shapes["u"])*.1
            v = rng.normal(size=ops.layout.shapes["v"])*.1
            if boundary == "free-slip":
                v[[0, -1]] = 0
            c = rng.uniform(.05, .5, (3, cfg.n, cfg.n))
            alpha = ops.layout.pack(u/3, v/3, c/5)
            result = ops.layout.unpack(ops.predictor(alpha))
            expected = reacting_dns.species_rhs(u, v, c, dns_config(cfg), cfg.damkohler)
            for i in range(3):
                np.testing.assert_allclose(5*result[f"c{i+1}"], expected[i], rtol=2e-14, atol=2e-14)
            self.assertLess(abs((result["c1"]+result["c3"]).sum()), 1e-14)
            self.assertLess(abs((result["c2"]+result["c3"]).sum()), 1e-14)
            for stage in (ops.pressure, ops.correction):
                np.testing.assert_array_equal(stage(alpha)[ops.layout.scalar_slice], 0)

    def test_reaction_decoupling_includes_annihilation_and_identity(self):
        rate = 400.  # Da * scalar encoding scale
        alpha = np.array([.2+.01j, .1-.02j, .03+.04j])
        gen = CoherentGenerator(3, {}, {(0, 0, 1): -rate, (1, 0, 1): -rate, (2, 0, 1): rate})
        f, b, c = gen.local_coefficients(alpha)
        common = rate*(-alpha[0].conjugate()-alpha[1].conjugate()+alpha[2].conjugate())
        np.testing.assert_allclose(f, rate*alpha[0]*alpha[1]*np.array([-1, -1, 1]))
        np.testing.assert_allclose(b, [common*alpha[1], common*alpha[0], 0])
        np.testing.assert_allclose(c, -alpha*b)

    def test_uniform_reaction_matches_analytic_solution_all_damkohler(self):
        for da in (1, 10, 100):
            cfg = config(damkohler=da)
            b, ops, _ = setup(cfg)
            c = np.stack([np.full((4, 4), .5), np.full((4, 4), .5), np.zeros((4, 4))])
            states = b.coherent_states(ops.layout.pack(np.zeros((4, 4)), np.zeros((5, 4)), c/4))
            # Zero velocity, uniform scalars: the complete predictor is pure reaction.
            for _ in range(20):
                states = b.advance(states, .001, ops.predictor, 8)
            result = mf.read_fields(states, b, ops, cfg)
            exact = .5/(1+da*.5*.02)
            np.testing.assert_allclose(result["c1"], exact, atol=2e-11, rtol=0)
            np.testing.assert_allclose(result["c2"], exact, atol=2e-11, rtol=0)
            np.testing.assert_allclose(result["c3"], .5-exact, atol=2e-11, rtol=0)

    def test_dns_uniform_reaction_is_second_order_and_conservative(self):
        cfg = dns_config(config())
        errors = []
        for dt in (.001, .0005, .00025):
            u, v = np.zeros((4, 4)), np.zeros((5, 4))
            c = np.stack([np.full((4, 4), .5), np.full((4, 4), .5), np.zeros((4, 4))])
            for _ in range(round(.02/dt)):
                u, v, _, c = reacting_dns.advance_one_step(u, v, c, dt, cfg, 100)
            errors.append(abs(c[0, 0, 0]-.25))
            np.testing.assert_allclose(c[0]+c[2], .5, atol=1e-15)
            np.testing.assert_allclose(c[1]+c[2], .5, atol=1e-15)
        self.assertGreater(errors[0]/errors[1], 3.8)
        self.assertGreater(errors[1]/errors[2], 3.8)

    def test_zero_reaction_recovers_passive_dns(self):
        cfg = config(damkohler=0)
        physical = dns_config(cfg)
        b, ops, states = setup(cfg)
        fields = mf.read_fields(states, b, ops, cfg)
        c = np.stack([fields[f"c{i}"] for i in (1, 2, 3)])
        u, v = fields["u"], fields["v"]
        actual = reacting_dns.advance_one_step(u, v, c, cfg.dt, physical, 0)
        for i in range(3):
            expected = dns.advance_one_step_with_scalar(u, v, c[i], cfg.dt, physical)
            for a, e in zip(actual[:3], expected[:3]):
                np.testing.assert_array_equal(a, e)
            np.testing.assert_array_equal(actual[3][i], expected[3])

    def test_all_stages_still_evolve_explicit_single_site_operators(self):
        cfg = config()
        b, ops, states = setup(cfg)
        used = set()
        original = b.derivative
        def record(kets, generator):
            used.add(generator)
            return original(kets, generator)
        reference = mf.scalar_references(mf.read_fields(states, b, ops, cfg), cfg)
        with patch.object(b, "derivative", side_effect=record):
            result, stats = mf.advance_one_step(states, cfg, b, ops)
        self.assertEqual(used, {ops.predictor, ops.pressure, ops.correction})
        mf.check_quality(mf.diagnostics(result, cfg, b, ops, reference, stats), cfg)

    def test_reaction_does_not_feed_back_on_velocity(self):
        results = []
        for da in (0, 100):
            cfg = config(damkohler=da)
            b, ops, states = setup(cfg)
            states, _ = mf.advance_one_step(states, cfg, b, ops)
            results.append(mf.read_fields(states, b, ops, cfg))
        for name in ("u", "v", "phi"):
            np.testing.assert_allclose(results[0][name], results[1][name], atol=1e-12, rtol=0)

    def test_restart_validation_and_three_species_dns_matching(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = mf.run_simulation(cfg, root/"direct")
            mf.run_simulation(cfg, root/"resumed", max_steps=3)
            resumed = mf.run_simulation(cfg, root/"resumed", resume=True)
            with np.load(direct) as a, np.load(resumed) as b:
                for key in ("local_states", "c1", "c2", "c3", "mass_reference"):
                    np.testing.assert_array_equal(a[key], b[key])
                corrupt = {key: a[key].copy() for key in a.files}
            self.assertTrue(mf.validate_results(resumed)["validated"])
            corrupt["c2"][-1, 0, 0] += .01
            mf.atomic_npz(root/"corrupt.npz", **corrupt)
            with self.assertRaisesRegex(ValueError, "c2 does not match"):
                mf.validate_results(root/"corrupt.npz")
            run_reference(direct, root/"dns.npz")
            first, second = load_fields(direct), load_fields(root/"dns.npz")
            self.assertTrue(identical_initial_fields(first, second))
            self.assertTrue(all(value == 0 for value in comparison(first, second)[0].values()))
            verify_dns_setup(cfg, second)
            with self.assertRaisesRegex(ValueError, "damkohler"):
                verify_dns_setup(replace(cfg, damkohler=10), second)
            self.assertGreater(first["c3"][-1].mean(), 0)
            self.assertLess(first["c1"][-1].mean(), .5)

    def test_reactive_postprocess_produces_requested_species_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = mf.run_simulation(config(damkohler=10), root/"run")
            summary = postprocess(result, root/"plots")
            self.assertTrue(summary["references"]["DNS"]["identical_initial_fields"])
            for species in ("c1", "c2", "c3"):
                for prefix in ("mean_field", "dns", "mean_field_vs_dns"):
                    self.assertGreater((root/"plots"/f"{prefix}_{species}.png").stat().st_size, 1000)
            report = (root/"plots"/"RUN_REPORT.md").read_text()
            self.assertIn("Da=10", report)
            self.assertIn("not positivity-preserving", report)
            self.assertIn("Positive and negative concentrations", report)
            self.assertEqual(summary["references"]["DNS"]["metadata"]["time_integrator"], "rk4")
            self.assertTrue((root/"plots"/"concentration_sign_history.csv").is_file())
            self.assertTrue((root/"plots"/"concentration_signs.png").is_file())
            self.assertEqual(json.loads((root/"plots"/"postprocess_status.json").read_text())["state"], "complete")

    def test_configuration_rejects_invalid_reactions(self):
        for changes in ({"scalar_species": 2}, {"damkohler": -1}, {"damkohler": np.nan},
                        {"scalar_species": 1, "damkohler": 1}):
            with self.assertRaises(ValueError):
                config(**changes)

    def test_negative_species_are_reported_without_clipping_or_false_mass_failure(self):
        cfg = config()
        # Conservative but nonphysical fields: conservation is not positivity.
        fields = {"c1": np.full((4, 4), -.01), "c2": np.full((4, 4), .4),
                  "c3": np.full((4, 4), .02)}
        reference = mf.scalar_references(fields, cfg)
        row = mf.scalar_diagnostics(fields, cfg, reference)
        self.assertLess(row["c1_minimum"], 0)
        self.assertLess(row["reaction_rate_minimum"], 0)
        self.assertEqual(row["reaction_negative_rate_fraction"], 1)
        self.assertEqual(row["scalar_mass_error"], 0)
        species = np.stack([fields[name] for name in ("c1", "c2", "c3")])
        independent = reacting_dns.conservation_diagnostics(species, reference)
        self.assertEqual(independent["c1_minimum"], row["c1_minimum"])
        np.testing.assert_array_equal(species[0], -.01)


if __name__ == "__main__":
    unittest.main()
