"""Literal forward-Euler integration, without hidden RK substeps or resets."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

import mixing_layer_dns as dns
import mixing_layer_mean_field as mf
import reacting_dns
from compare_mean_field_dns import dns_config, run_reference
from mean_field_bosons import LocalBosons
from mean_field_operators import CoherentGenerator, MeanFieldOperators
from plot_mean_field_results import identical_initial_fields, load_fields, verify_dns_setup


class EulerTests(unittest.TestCase):
    def test_literal_normalized_ket_update_and_one_derivative_per_substep(self):
        bosons = LocalBosons(6)
        states = bosons.normalize(np.array([[1, .2, -.04, .01, 0, 0, 0]], float))
        generator = CoherentGenerator(1, {(0, 0): -.3}, {(0, 0, 0): .2})
        expected = bosons.normalize(states+.01*bosons.derivative(states, generator))
        with patch.object(bosons, "derivative", wraps=bosons.derivative) as derivative:
            actual = bosons.advance(states, .01, generator, method="forward-euler")
        self.assertEqual(derivative.call_count, 1)
        np.testing.assert_array_equal(actual, expected)
        with patch.object(bosons, "derivative", wraps=bosons.derivative) as derivative:
            bosons.advance(states, .01, generator, 8, method="forward-euler")
        self.assertEqual(derivative.call_count, 8)

    def test_first_order_convergence_to_fixed_operator_solution(self):
        bosons = LocalBosons(6)
        states = bosons.normalize(np.array([[1, .2, -.04, .01, 0, 0, 0]], float))
        class Fixed:
            @staticmethod
            def local_coefficients(alpha):
                return np.array([.7]), np.array([-.2]), np.array([.1])
        matrix = .7*bosons.creation-.2*bosons.annihilation+.1*bosons.identity
        exact = bosons.normalize((expm(.2*matrix)@states[0])[None, :])
        errors = [np.linalg.norm(bosons.advance(states, .2, Fixed(), k, method="forward-euler")-exact)
                  for k in (20, 40, 80)]
        for ratio in (errors[0]/errors[1], errors[1]/errors[2]):
            self.assertGreater(ratio, 1.9)
            self.assertLess(ratio, 2.1)

    def test_all_mean_field_stages_select_euler(self):
        cfg = mf.MeanFieldConfig(n=4, final_time=.0175, kh_amplitude=.2, secondary_amplitude=.03,
                                 scalar_species=3, damkohler=1, time_integrator="forward-euler")
        bosons, ops = LocalBosons(cfg.boson_cutoff), MeanFieldOperators(cfg)
        states = bosons.coherent_states(ops.layout.pack(*mf.initial_amplitudes(cfg)))
        with patch.object(bosons, "advance", wraps=bosons.advance) as advance:
            mf.advance_one_step(states, cfg, bosons, ops)
        generators = set()
        for call in advance.call_args_list:
            generators.add(call.args[2])
            self.assertEqual(call.kwargs["method"], "forward-euler")
        self.assertEqual(generators, {ops.predictor, ops.pressure, ops.correction})

    def test_dns_is_single_rhs_evaluation_followed_by_projection(self):
        cfg = mf.MeanFieldConfig(n=4, final_time=.0175, scalar_species=3, damkohler=10,
                                 kh_amplitude=.2, secondary_amplitude=.03, time_integrator="forward-euler")
        physical = dns_config(cfg)
        au, av, ac = mf.initial_amplitudes(cfg)
        u, v, c = cfg.velocity_scale*au, cfg.velocity_scale*av, cfg.scalar_scale*ac
        ru, rv = dns.channel_momentum_rhs(u, v, physical.viscosity, .25, .25)
        expected_c = c+cfg.dt*reacting_dns.species_rhs(u, v, c, physical, 10)
        expected_velocity = dns.project_channel_velocity(u+cfg.dt*ru, v+cfg.dt*rv, cfg.dt, .25, .25)
        with patch.object(reacting_dns, "species_rhs", wraps=reacting_dns.species_rhs) as rhs:
            actual = reacting_dns.advance_one_step(u, v, c, cfg.dt, physical, 10, method="forward-euler")
        self.assertEqual(rhs.call_count, 1)
        for value, expected in zip(actual[:3], expected_velocity):
            np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(actual[3], expected_c)

    def test_dns_uniform_reaction_has_first_order_convergence(self):
        physical = dns_config(mf.MeanFieldConfig(n=4))
        errors = []
        for dt in (.001, .0005, .00025):
            u, v = np.zeros((4, 4)), np.zeros((5, 4))
            c = np.stack([np.full((4, 4), .5), np.full((4, 4), .5), np.zeros((4, 4))])
            for _ in range(round(.02/dt)):
                u, v, _, c = reacting_dns.advance_one_step(u, v, c, dt, physical, 100, method="forward-euler")
            errors.append(abs(c[0, 0, 0]-.25))
            np.testing.assert_allclose(c[0]+c[2], .5, atol=1e-15)
        self.assertGreater(errors[0]/errors[1], 1.9)
        self.assertLess(errors[0]/errors[1], 2.2)
        self.assertGreater(errors[1]/errors[2], 1.9)

    def test_configuration_and_dns_matching_reject_wrong_integrators(self):
        with self.assertRaises(ValueError):
            mf.MeanFieldConfig(time_integrator="midpoint")
        with self.assertRaises(ValueError):
            mf.MeanFieldConfig(time_integrator="forward-euler", pressure_cfl=.26)
        cfg = mf.MeanFieldConfig(time_integrator="forward-euler")
        reference = {"config": {**asdict(dns_config(cfg)), "time_integrator": "forward-euler"}}
        verify_dns_setup(cfg, reference)
        with self.assertRaisesRegex(ValueError, "time_integrator"):
            verify_dns_setup(replace(cfg, time_integrator="rk4"), reference)

    def test_small_euler_trajectory_and_reference_preserve_method_metadata(self):
        cfg = mf.MeanFieldConfig(n=4, dt=.0001, final_time=.0007, kh_amplitude=.02,
            secondary_amplitude=.003, scalar_species=3, damkohler=1,
            reynolds=100, peclet=100, time_integrator="forward-euler")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = mf.run_simulation(cfg, root/"mf")
            validation = mf.validate_results(result)
            self.assertEqual(validation["run_metadata"]["time_integrator"], "forward-euler")
            self.assertIn("forward-euler", validation["run_metadata"]["pressure_method"])
            run_reference(result, root/"dns.npz")
            primary, reference = load_fields(result), load_fields(root/"dns.npz")
            self.assertTrue(identical_initial_fields(primary, reference))
            self.assertEqual(reference["run_metadata"]["time_integrator"], "forward-euler")
            verify_dns_setup(cfg, reference)

    def test_pressure_observer_records_convergence_without_changing_dynamics(self):
        cfg = mf.MeanFieldConfig(n=4, final_time=.0175, kh_amplitude=.2, secondary_amplitude=.03,
                                 scalar_species=3, damkohler=1, time_integrator="forward-euler")
        bosons, ops = LocalBosons(cfg.boson_cutoff), MeanFieldOperators(cfg)
        states = bosons.coherent_states(ops.layout.pack(*mf.initial_amplitudes(cfg)))
        baseline, stats = mf.advance_one_step(states, cfg, bosons, ops)
        history = []
        observed, recorded = mf.advance_one_step(states, cfg, bosons, ops, pressure_observer=history.append)
        np.testing.assert_array_equal(observed, baseline)
        self.assertEqual(recorded, stats)
        self.assertEqual(history[0]["pressure_iterations"], 0)
        self.assertEqual(history[-1], {key: stats[key] for key in
            ("pressure_iterations", "pressure_residual", "pressure_pseudo_time")})
        self.assertLessEqual(history[-1]["pressure_residual"], cfg.pressure_tolerance)

    def test_reacting_euler_checkpoint_resume_preserves_all_local_kets(self):
        cfg = mf.MeanFieldConfig(n=4, dt=.0001, final_time=.0007, kh_amplitude=.02,
            secondary_amplitude=.003, scalar_species=3, damkohler=100,
            reynolds=200, peclet=200, time_integrator="forward-euler")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = mf.run_simulation(cfg, root/"uninterrupted", checkpoint_interval=1)
            mf.run_simulation(cfg, root/"resumed", max_steps=3, checkpoint_interval=1)
            resumed = mf.run_simulation(cfg, root/"resumed", resume=True, checkpoint_interval=1)
            with np.load(complete, allow_pickle=False) as first, np.load(resumed, allow_pickle=False) as second:
                for key in ("terminal_local_states", "local_states", "history_json", "fingerprint"):
                    np.testing.assert_array_equal(first[key], second[key])
            self.assertTrue(mf.validate_results(resumed)["validated"])


if __name__ == "__main__":
    unittest.main()
