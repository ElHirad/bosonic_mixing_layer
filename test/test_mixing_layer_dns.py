"""Numerical consistency tests for the periodic staggered-grid solver."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from mixing_layer_dns import (
    SimulationConfig,
    divergence,
    flow_diagnostics,
    initialize_velocity,
    laplacian,
    momentum_rhs,
    periodic_double_tanh_profile,
    pressure_gradient,
    project_velocity,
    run_simulation,
    solve_periodic_poisson,
)


class MACOperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dx = 0.2
        self.dy = 0.3
        self.shape = (3, 5)
        self.rng = np.random.default_rng(20260817)

    def test_gradient_and_divergence_are_negative_adjoints(self) -> None:
        pressure = self.rng.standard_normal(self.shape)
        u = self.rng.standard_normal(self.shape)
        v = self.rng.standard_normal(self.shape)
        grad_x, grad_y = pressure_gradient(pressure, self.dx, self.dy)
        lhs = self.dx * self.dy * np.sum(
            pressure * divergence(u, v, self.dx, self.dy)
        )
        rhs = -self.dx * self.dy * np.sum(grad_x * u + grad_y * v)
        self.assertAlmostEqual(float(lhs), float(rhs), places=12)

    def test_laplacian_matches_divergence_of_gradient(self) -> None:
        field = self.rng.standard_normal(self.shape)
        grad_x, grad_y = pressure_gradient(field, self.dx, self.dy)
        composed = divergence(grad_x, grad_y, self.dx, self.dy)
        np.testing.assert_allclose(
            composed, laplacian(field, self.dx, self.dy), rtol=2e-14, atol=2e-14
        )

    def test_periodic_poisson_manufactured_solution(self) -> None:
        expected = self.rng.standard_normal(self.shape)
        expected -= np.mean(expected)
        rhs = laplacian(expected, self.dx, self.dy)
        actual = solve_periodic_poisson(rhs, self.dx, self.dy)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        self.assertAlmostEqual(float(np.mean(actual)), 0.0, places=14)

    def test_projection_removes_divergence_and_preserves_means(self) -> None:
        u_star = self.rng.standard_normal(self.shape)
        v_star = self.rng.standard_normal(self.shape)
        u, v, pressure = project_velocity(u_star, v_star, 0.017, self.dx, self.dy)
        self.assertLess(np.max(np.abs(divergence(u, v, self.dx, self.dy))), 2e-13)
        self.assertAlmostEqual(float(np.mean(u)), float(np.mean(u_star)), places=13)
        self.assertAlmostEqual(float(np.mean(v)), float(np.mean(v_star)), places=13)
        self.assertAlmostEqual(float(np.mean(pressure)), 0.0, places=13)
        energy_before = np.mean(u_star**2 + v_star**2)
        energy_after = np.mean(u**2 + v**2)
        self.assertLessEqual(energy_after, energy_before * (1.0 + 2e-14))

    def test_projection_rejects_a_pure_pressure_gradient(self) -> None:
        potential = self.rng.standard_normal(self.shape)
        u_star, v_star = pressure_gradient(potential, self.dx, self.dy)
        u, v, _ = project_velocity(u_star, v_star, 0.1, self.dx, self.dy)
        self.assertLess(np.max(np.abs(u)), 2e-13)
        self.assertLess(np.max(np.abs(v)), 2e-13)


class MixingLayerTests(unittest.TestCase):
    def small_config(self) -> SimulationConfig:
        return SimulationConfig(
            nx=32,
            ny=32,
            dx=1.0 / 32.0,
            dy=1.0 / 32.0,
            transition_thickness=0.06,
            perturbation_width=0.12,
            max_dt=0.001,
            t_end=0.007,
        )

    def test_profile_has_requested_layer_signs_and_exact_periodicity(self) -> None:
        config = SimulationConfig()
        y = np.array([0.0, 0.125, 0.5, 0.875, 1.0])
        profile = periodic_double_tanh_profile(y, config)
        self.assertLess(profile[1], -0.99)
        self.assertGreater(profile[2], 0.99)
        self.assertLess(profile[3], -0.99)
        self.assertAlmostEqual(float(profile[0]), float(profile[-1]), places=14)

    def test_default_middle_layer_is_thirty_percent_of_domain(self) -> None:
        config = SimulationConfig()
        lower = 0.5 * (1.0 - config.middle_layer_fraction) * config.ly
        upper = 0.5 * (1.0 + config.middle_layer_fraction) * config.ly
        self.assertAlmostEqual(lower, 0.35)
        self.assertAlmostEqual(upper, 0.65)

    def test_streamfunction_initialization_is_discretely_solenoidal(self) -> None:
        config = SimulationConfig()
        u, v, _ = initialize_velocity(config)
        div_max = np.max(np.abs(divergence(u, v, config.dx, config.dy)))
        self.assertLess(div_max, 2e-13)

        x_mean_u = np.mean(u, axis=1)
        y_faces = (np.arange(config.ny) + 0.5) * config.dy
        expected = periodic_double_tanh_profile(y_faces, config)
        np.testing.assert_allclose(x_mean_u, expected, rtol=1e-12, atol=1e-12)

    def test_centered_advection_has_no_inviscid_energy_production(self) -> None:
        config = self.small_config()
        u, v, _ = initialize_velocity(config)
        rhs_u, rhs_v = momentum_rhs(u, v, 0.0, config.dx, config.dy)
        energy_rate = float(np.mean(u * rhs_u + v * rhs_v))
        self.assertLess(abs(energy_rate), 2e-13)

    def test_short_run_returns_exactly_eight_finite_snapshots(self) -> None:
        config = self.small_config()
        snapshots = run_simulation(config)
        self.assertEqual(len(snapshots), 8)
        np.testing.assert_allclose(
            [snapshot.time for snapshot in snapshots],
            np.linspace(0.0, config.t_end, 8),
            rtol=0.0,
            atol=2e-15,
        )
        for snapshot in snapshots:
            self.assertTrue(np.all(np.isfinite(snapshot.u)))
            self.assertTrue(np.all(np.isfinite(snapshot.v)))
            self.assertLess(snapshot.diagnostics["max_divergence"], 5e-12)

        initial_energy = snapshots[0].diagnostics["kinetic_energy"]
        final_energy = snapshots[-1].diagnostics["kinetic_energy"]
        self.assertLessEqual(final_energy, initial_energy * (1.0 + 1e-10))


if __name__ == "__main__":
    unittest.main()
