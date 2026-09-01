"""Numerical consistency tests for the staggered-grid DNS solvers."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from mixing_layer_dns import (
    SimulationConfig,
    advance_one_step,
    advance_one_step_with_scalar,
    channel_divergence,
    channel_momentum_rhs,
    channel_neumann_laplacian,
    channel_pressure_gradient,
    conservative_scalar_advection,
    divergence,
    flow_diagnostics,
    initialize_velocity,
    initialize_concentration,
    laplacian,
    momentum_rhs,
    periodic_double_tanh_profile,
    pressure_gradient,
    project_velocity,
    project_channel_velocity,
    run_simulation,
    solve_periodic_poisson,
    solve_channel_poisson,
    single_tanh_profile,
    scalar_rhs,
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
            boundary_y="periodic",
            transition_thickness=0.06,
            perturbation_width=0.12,
            max_dt=0.001,
            t_end=0.007,
        )

    def test_profile_has_requested_layer_signs_and_exact_periodicity(self) -> None:
        config = SimulationConfig(boundary_y="periodic")
        y = np.array([0.0, 0.125, 0.5, 0.875, 1.0])
        profile = periodic_double_tanh_profile(y, config)
        self.assertLess(profile[1], -0.99)
        self.assertGreater(profile[2], 0.99)
        self.assertLess(profile[3], -0.99)
        self.assertAlmostEqual(float(profile[0]), float(profile[-1]), places=14)

    def test_default_middle_layer_is_thirty_percent_of_domain(self) -> None:
        config = SimulationConfig(boundary_y="periodic")
        lower = 0.5 * (1.0 - config.middle_layer_fraction) * config.ly
        upper = 0.5 * (1.0 + config.middle_layer_fraction) * config.ly
        self.assertAlmostEqual(lower, 0.35)
        self.assertAlmostEqual(upper, 0.65)

    def test_streamfunction_initialization_is_discretely_solenoidal(self) -> None:
        config = SimulationConfig(boundary_y="periodic")
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

    def test_scalar_flux_and_diffusion_conserve_periodic_mass(self) -> None:
        config = self.small_config()
        u, v, _ = initialize_velocity(config)
        concentration = np.random.default_rng(20260831).standard_normal(
            (config.ny, config.nx)
        )
        advection = conservative_scalar_advection(
            u, v, concentration, config.dx, config.dy
        )
        rhs = scalar_rhs(
            u,
            v,
            concentration,
            config.scalar_diffusivity,
            config.dx,
            config.dy,
        )
        self.assertLess(abs(float(np.sum(advection))), 2e-12)
        self.assertLess(abs(float(np.sum(rhs))), 2e-12)

    def test_initial_concentration_is_exact_zero_one_double_tanh(self) -> None:
        config = self.small_config()
        concentration = initialize_concentration(config)
        self.assertEqual(concentration.shape, (config.ny, config.nx))
        self.assertAlmostEqual(float(np.min(concentration)), 0.0, places=14)
        self.assertAlmostEqual(float(np.max(concentration)), 1.0, places=14)
        np.testing.assert_allclose(
            concentration,
            np.repeat(concentration[:, :1], config.nx, axis=1),
            rtol=0.0,
            atol=0.0,
        )

    def test_coupled_midpoint_step_matches_velocity_only_and_conserves_scalar(self) -> None:
        config = self.small_config()
        u, v, _ = initialize_velocity(config)
        concentration = initialize_concentration(config)
        dt = 0.5 * config.max_dt
        expected_u, expected_v, expected_pressure = advance_one_step(u, v, dt, config)
        actual_u, actual_v, actual_pressure, actual_concentration = (
            advance_one_step_with_scalar(u, v, concentration, dt, config)
        )
        np.testing.assert_allclose(actual_u, expected_u, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual_v, expected_v, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            actual_pressure, expected_pressure, rtol=0.0, atol=0.0
        )
        self.assertAlmostEqual(
            float(np.mean(actual_concentration)),
            float(np.mean(concentration)),
            places=14,
        )
        self.assertTrue(np.all(np.isfinite(actual_concentration)))

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


class SingleLayerChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nx = 9
        self.ny = 7
        self.dx = 0.17
        self.dy = 0.13
        self.rng = np.random.default_rng(20260901)

    def test_channel_gradient_divergence_matches_neumann_laplacian(self) -> None:
        pressure = self.rng.standard_normal((self.ny, self.nx))
        grad_x, grad_y = channel_pressure_gradient(pressure, self.dx, self.dy)
        composed = channel_divergence(grad_x, grad_y, self.dx, self.dy)
        expected = channel_neumann_laplacian(pressure, self.dx, self.dy)
        np.testing.assert_allclose(composed, expected, rtol=3e-14, atol=3e-14)
        np.testing.assert_array_equal(grad_y[[0, -1], :], 0.0)

    def test_channel_poisson_manufactured_solution(self) -> None:
        expected = self.rng.standard_normal((self.ny, self.nx))
        expected -= np.mean(expected)
        rhs = channel_neumann_laplacian(expected, self.dx, self.dy)
        actual = solve_channel_poisson(rhs, self.dx, self.dy)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def test_channel_projection_removes_divergence_and_keeps_walls_impermeable(self) -> None:
        u_star = self.rng.standard_normal((self.ny, self.nx))
        v_star = self.rng.standard_normal((self.ny + 1, self.nx))
        u, v, pressure = project_channel_velocity(
            u_star, v_star, 0.011, self.dx, self.dy
        )
        self.assertLess(
            float(np.max(np.abs(channel_divergence(u, v, self.dx, self.dy)))),
            4e-13,
        )
        np.testing.assert_array_equal(v[[0, -1], :], 0.0)
        self.assertAlmostEqual(float(np.mean(pressure)), 0.0, places=13)

    def test_single_layer_initialization_and_short_run(self) -> None:
        config = SimulationConfig(
            nx=64,
            ny=64,
            dx=1.0 / 64.0,
            dy=1.0 / 64.0,
            transition_thickness=0.04,
            perturbation_width=0.10,
            max_dt=0.001,
            t_end=0.007,
        )
        u, v, _ = initialize_velocity(config)
        self.assertEqual(u.shape, (64, 64))
        self.assertEqual(v.shape, (65, 64))
        self.assertLess(
            float(np.max(np.abs(channel_divergence(u, v, config.dx, config.dy)))),
            2e-13,
        )
        np.testing.assert_array_equal(v[[0, -1], :], 0.0)
        y = np.array([0.1, 0.5, 0.9])
        profile = single_tanh_profile(y, config)
        self.assertLess(profile[0], -0.999)
        self.assertAlmostEqual(float(profile[1]), 0.0, places=14)
        self.assertGreater(profile[2], 0.999)

        rhs_u, rhs_v = channel_momentum_rhs(u, v, 0.0, config.dx, config.dy)
        energy_rate = float(np.sum(u * rhs_u) + np.sum(v[1:-1] * rhs_v[1:-1]))
        self.assertLess(abs(energy_rate), 2e-10)

        snapshots = run_simulation(config)
        self.assertEqual(len(snapshots), 8)
        for snapshot in snapshots:
            self.assertLess(snapshot.diagnostics["max_divergence"], 2e-11)
            np.testing.assert_array_equal(snapshot.v[[0, -1], :], 0.0)


if __name__ == "__main__":
    unittest.main()
