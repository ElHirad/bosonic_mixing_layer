"""Explicit single-site MF algebra, boundary stencils, dynamics, and restart.

Classical routines appear ONLY as independent test oracles, never in the
single-site production solver or its pressure/correction stages.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

import mixing_layer_dns as dns
import mixing_layer_mean_field as mf
from mean_field_bosons import LocalBosons, relax_pressure
from mean_field_operators import CoherentGenerator, MeanFieldOperators


def small_config(**changes):
    return replace(mf.MeanFieldConfig(n=4, kh_amplitude=0.2, secondary_amplitude=0.03,
                                    final_time=0.0175), **changes)


def setup_case(config):
    bosons, operators = LocalBosons(config.boson_cutoff), MeanFieldOperators(config)
    alpha = operators.layout.pack(*mf.initial_amplitudes(config))
    return bosons, operators, bosons.coherent_states(alpha)


class SingleSiteAlgebraTests(unittest.TestCase):
    def test_local_integrator_converges_to_explicit_matrix_exponential(self):
        bosons = LocalBosons(5)
        state = bosons.normalize(np.array([[1., .1, -.03, .04, 0., .02]]))
        class FixedMeanField:
            @staticmethod
            def local_coefficients(alpha):
                return np.array([.7]), np.array([-.2]), np.array([.11])
        matrix = .7*bosons.creation-.2*bosons.annihilation+.11*bosons.identity
        exact = bosons.normalize((expm(.5*matrix)@state[0])[None, :])
        errors = [np.linalg.norm(bosons.advance(state, .5, FixedMeanField(), count)-exact)
                  for count in (4, 8, 16)]
        self.assertGreater(errors[0]/errors[1], 12)
        self.assertGreater(errors[1]/errors[2], 12)
        self.assertLess(errors[-1], 1e-8)

    def test_explicit_operators_and_truncated_commutator(self):
        b = LocalBosons(6)
        np.testing.assert_array_equal(b.creation, b.annihilation.T)
        expected = np.eye(7)
        expected[-1, -1] -= 7
        np.testing.assert_allclose(b.annihilation@b.creation-b.creation@b.annihilation, expected, atol=2e-15)

    def test_linear_and_cubic_decoupling_includes_all_three_local_terms(self):
        alpha = np.array([0.2+0.1j, -0.3+0.2j])
        # 3 a_0^dagger a_1 + 2 a_1^dagger a_0 a_0
        gen = CoherentGenerator(2, {(0, 1): 3}, {(1, 0, 0): 2})
        f, b, c = gen.local_coefficients(alpha)
        np.testing.assert_allclose(f, [3*alpha[1], 2*alpha[0]**2])
        np.testing.assert_allclose(b, [4*alpha[1].conjugate()*alpha[0], 3*alpha[0].conjugate()])
        np.testing.assert_allclose(c.sum(), -3*alpha[0].conjugate()*alpha[1]
                                   -4*alpha[1].conjugate()*alpha[0]**2)

    def test_derivative_is_explicit_matrix_action_even_for_noncoherent_kets(self):
        b = LocalBosons(5)
        states = b.normalize(np.array([[1, .2, .1, 0, .03, 0], [1, -.1, 0, .02, 0, 0]], float))
        gen = CoherentGenerator(2, {(0, 1): .7, (1, 1): -.3}, {(1, 0, 0): .2})
        f, lower, c = gen.local_coefficients(b.amplitudes(states))
        expected = []
        for j, state in enumerate(states):
            matrix = f[j]*b.creation+lower[j]*b.annihilation+c[j]*b.identity
            expected.append(matrix@state-np.vdot(state, matrix@state).real*state)
        np.testing.assert_allclose(b.derivative(states, gen), expected, atol=1e-16)
        self.assertGreater(b.quality(states)["coherent_eigenstate_defect"], 0.01)

    def test_coherent_expectation_derivative_recovers_generator_coefficients(self):
        b = LocalBosons(16)
        alpha = np.array([.2+.04j, -.1+.02j])
        states = b.coherent_states(alpha)
        gen = CoherentGenerator(2, {(0, 1): .7, (1, 1): -.3}, {(1, 0, 0): .2})
        rate = b.derivative(states, gen)
        dalpha = np.sum(rate.conj()*(states@b.annihilation.T)
                       + states.conj()*(rate@b.annihilation.T), axis=1)
        np.testing.assert_allclose(dalpha, gen(alpha), rtol=1e-13, atol=1e-15)

    def test_finite_cutoff_convergence_and_old_cutoff_rejection(self):
        defects = []
        for cutoff in (4, 8, 12):
            cfg = mf.MeanFieldConfig(boson_cutoff=cutoff)
            b, ops, states = setup_case(cfg)
            defects.append(b.quality(states)["coherent_eigenstate_defect"])
            if cutoff == 4:
                with self.assertRaises(FloatingPointError):
                    mf.check_quality(mf.diagnostics(states, cfg, b, ops, .5), cfg)
        self.assertGreater(defects[0], 1e-4)
        self.assertLess(defects[1], defects[0]/1000)
        self.assertLess(defects[2], 1e-10)


class MeanFieldStencilTests(unittest.TestCase):
    def test_all_generator_coefficients_against_independent_stencils(self):
        rng = np.random.default_rng(71)
        for boundary in ("free-slip", "periodic"):
            cfg = small_config(boundary_y=boundary, velocity_scale=3, pressure_scale=7, scalar_scale=5)
            b, ops, _ = setup_case(cfg)
            fields = {f: rng.normal(size=shape)*.1 for f, shape in ops.layout.shapes.items()}
            if boundary == "free-slip":
                fields["v"][[0, -1]] = 0
            alpha = ops.layout.pack(fields["u"]/3, fields["v"]/3, fields["c"]/5, fields["phi"]/7)
            u, v, c, p = (fields[f] for f in ("u", "v", "c", "phi"))
            h = 1/cfg.n
            if boundary == "free-slip":
                ru, rv = dns.channel_momentum_rhs(u, v, h*0+.02, h, h)
                grad = dns.channel_pressure_gradient
                lap = dns.channel_neumann_laplacian
                div = dns.channel_divergence
            else:
                ru, rv = dns.momentum_rhs(u, v, .02, h, h)
                grad, lap, div = dns.pressure_gradient, dns.laplacian, dns.divergence
            rc = dns.scalar_rhs(u, v, c, .02, h, h)
            pred = ops.layout.unpack(ops.predictor(alpha))
            for field, expected, scale in (("u", ru, 3), ("v", rv, 3), ("c", rc, 5)):
                np.testing.assert_allclose(pred[field]*scale, expected, rtol=2e-14, atol=2e-14)
            self.assertLess(abs(pred["c"].sum()), 2e-15)
            pressure = ops.layout.unpack(ops.pressure(alpha))
            np.testing.assert_allclose(7*pressure["phi"], lap(p, h, h)-div(u, v, h, h), atol=2e-14)
            corr = ops.layout.unpack(ops.correction(alpha))
            gx, gy = grad(p, h, h)
            np.testing.assert_allclose(3*corr["u"], -gx, atol=1e-15)
            np.testing.assert_allclose(3*corr["v"], -gy, atol=1e-15)
            for field in ("phi", "c"):
                np.testing.assert_array_equal(corr[field], 0)

    def test_initialization_matches_last_case_without_projection(self):
        for boundary in ("free-slip", "periodic"):
            cfg = mf.MeanFieldConfig(boundary_y=boundary)
            pc = dns.SimulationConfig(nx=16, ny=16, dx=1/16, dy=1/16, reynolds=50, peclet=50,
                boundary_y=boundary, transition_thickness=.04, perturbation_width=.12,
                perturbation_mode=2, perturbation_amplitude=2.5, subharmonic_mode=1,
                subharmonic_amplitude=.5, perturbation_phase=0)
            au, av, ac = mf.initial_amplitudes(cfg)
            u, v, _ = dns.initialize_velocity(pc)
            np.testing.assert_allclose(8*au, u, atol=1e-15)
            np.testing.assert_allclose(8*av, v, atol=1e-15)
            np.testing.assert_allclose(4*ac, dns.initialize_concentration(pc), atol=1e-15)
            self.assertLess(np.max(np.abs(mf.divergence(8*au, 8*av, cfg))), 3e-14)

    def test_local_pressure_relaxation_and_correction(self):
        cfg = small_config()
        b, ops, states = setup_case(cfg)
        fields = ops.layout.unpack(b.amplitudes(states))
        n, h = cfg.n, 1/cfg.n
        potential = .001*np.cos(2*np.pi*(np.arange(n)+.5)/n)[None, :]*np.ones((n, 1))
        gx, gy = dns.channel_pressure_gradient(potential, h, h)
        states = b.coherent_states(ops.layout.pack(gx/8, gy/8, fields["c"]))
        relaxed, stats = relax_pressure(states, b, ops, cfg.pseudo_dt, cfg.pressure_tolerance, cfg.pressure_max_steps)
        self.assertGreater(stats["pressure_iterations"], 0)
        physical = mf.read_fields(relaxed, b, ops, cfg)
        np.testing.assert_allclose(physical["phi"], potential, atol=2e-10)
        corrected = b.advance(relaxed, 1, ops.correction, 8)
        result = mf.read_fields(corrected, b, ops, cfg)
        self.assertLess(np.max(abs(result["u"])), 2e-9)
        np.testing.assert_allclose(result["c"], 4*fields["c"], atol=1e-15)

    def test_pressure_nonconvergence_is_an_error(self):
        cfg = small_config()
        b, ops, states = setup_case(cfg)
        star = b.advance(states, cfg.dt, ops.predictor, 4)
        with self.assertRaisesRegex(FloatingPointError, "failed after"):
            relax_pressure(star, b, ops, cfg.pseudo_dt, 1e-12, 1)

    def test_all_three_stages_use_explicit_single_site_derivative(self):
        cfg = small_config()
        b, ops, states = setup_case(cfg)
        used = set()
        derivative = b.derivative
        def record(states, generator):
            used.add(generator)
            return derivative(states, generator)
        with patch.object(b, "derivative", side_effect=record):
            result, stats = mf.advance_one_step(states, cfg, b, ops)
        self.assertEqual(used, {ops.predictor, ops.pressure, ops.correction})
        mf.check_quality(mf.diagnostics(result, cfg, b, ops, .5, stats), cfg)

    def test_concentration_is_passive_in_converged_local_state_evolution(self):
        cfg = small_config()
        b, ops, coupled = setup_case(cfg)
        flow_only = coupled.copy()
        flow_only[ops.layout.slices["c"]] = 0
        flow_only[ops.layout.slices["c"], 0] = 1
        coupled, _ = mf.advance_one_step(coupled, cfg, b, ops)
        flow_only, _ = mf.advance_one_step(flow_only, cfg, b, ops)
        first = mf.read_fields(coupled, b, ops, cfg)
        second = mf.read_fields(flow_only, b, ops, cfg)
        for field in ("u", "v", "phi"):
            np.testing.assert_allclose(first[field], second[field], rtol=0, atol=1e-11)


class MeanFieldPersistenceTests(unittest.TestCase):
    def test_complete_run_restart_and_independent_local_state_validation(self):
        cfg = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            direct = mf.run_simulation(cfg, Path(tmp)/"direct")
            mf.run_simulation(cfg, Path(tmp)/"resume", max_steps=3)
            resumed = mf.run_simulation(cfg, Path(tmp)/"resume", resume=True)
            with np.load(direct) as a, np.load(resumed) as b:
                for key in ("local_states", "terminal_local_states", "concentration", "u", "v"):
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(json.loads(str(b["run_metadata_json"]))["resumed_from_step"], 3)
            result = mf.validate_results(resumed)
            self.assertTrue(result["validated"])
            self.assertLess(result["worst_step"]["scalar_mass_error"], 1e-9)
            with self.assertRaises(FileExistsError):
                mf.run_simulation(cfg, Path(tmp)/"direct")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                mf.run_simulation(replace(cfg, peclet=51), Path(tmp)/"resume", resume=True)
            with np.load(direct) as data:
                corrupt = {key: data[key].copy() for key in data.files}
            corrupt["concentration"][-1, 0, 0] += .01
            path = Path(tmp)/"corrupt.npz"
            mf.atomic_npz(path, **corrupt)
            with self.assertRaisesRegex(ValueError, "does not match"):
                mf.validate_results(path)

    def test_stop_checkpoint_and_incomplete_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp)/"STOP"
            stop.touch()
            path = mf.run_simulation(small_config(), Path(tmp)/"run", stop_file=stop)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                mf.validate_results(path)
            with np.load(path) as data:
                self.assertEqual(int(data["completed_step"]), 0)

    def test_failed_step_keeps_last_accepted_local_states(self):
        cfg = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(mf, "advance_one_step", side_effect=FloatingPointError("test failure")):
                with self.assertRaises(FloatingPointError):
                    mf.run_simulation(cfg, tmp)
            self.assertEqual(json.loads((Path(tmp)/"run_status.json").read_text())["state"], "failed")
            with np.load(Path(tmp)/"checkpoint.npz") as checkpoint:
                self.assertEqual(int(checkpoint["completed_step"]), 0)
                np.testing.assert_array_equal(checkpoint["local_states"][0], checkpoint["terminal_local_states"])

    def test_invalid_configuration(self):
        for changes in ({"n": 3}, {"reynolds": 0}, {"dt": -.1}, {"final_time": .011},
                        {"pressure_cfl": .5}, {"boson_cutoff": 0}, {"velocity_scale": np.nan}):
            with self.assertRaises(ValueError):
                small_config(**changes)


if __name__ == "__main__":
    unittest.main()
